# Palier 2 : une mémoire qui se comprend et se corrige

## 1. Provenance et annulation des mémoires

Chaque commit de session écrit déjà `memory_diff.json` dans son archive
(`viking://user/{uid}/sessions/{sid}/history/archive_NNN/`) avec les mémoires
ajoutées, modifiées (avant/après) et supprimées. Le fork expose ces journaux :

| Endpoint | Rôle |
|----------|------|
| `GET /api/v1/memory/provenance?uri=viking://~/memories/...&limit=50` | quelles archives ont créé, modifié ou supprimé cette mémoire, avec le contenu avant et après |
| `POST /api/v1/memory/revert` `{"uri": ..., "archive_uri": ...}` | annule le changement enregistré : un ajout est supprimé, une modification retrouve son contenu précédent, une suppression est recréée |
| `GET /api/v1/memory/as-of?uri=...&at=2026-09-01T12:00:00Z` | le fichier tel qu'il était dans le dernier instantané validé à cette date |

L'annulation passe par les chemins d'écriture et de suppression habituels
(`FSService.write`, `FSService.rm`), donc vecteurs et fichiers annexes sont
rafraîchis. Chaque annulation est consignée dans `memory_reverts.json` à côté
du journal de l'archive.

```bash
curl "http://127.0.0.1:1933/api/v1/memory/provenance?uri=viking://~/memories/preferences/editor.md"
curl -X POST http://127.0.0.1:1933/api/v1/memory/revert -H "Content-Type: application/json" \
  -d '{"uri": "viking://~/memories/preferences/editor.md", "archive_uri": "viking://~/sessions/s1/history/archive_001"}'
```

Code : `openviking/service/memory_timeline.py`,
`openviking/server/routers/memory_timeline.py`. Tests :
`tests/service/test_memory_timeline.py`.

## 2. Mémoire temporelle

Les instantanés (gitoxide dans RAGFS) ne sont pris que sur demande en amont.
Avec `memory.snapshot_on_commit = true`, le fork enregistre automatiquement le
dossier `viking://~/memories` après chaque commit de session qui a changé au
moins une mémoire. L'endpoint `as-of` ci-dessus répond alors à « que savais-je
à telle date » en lisant le dernier instantané antérieur à l'instant demandé.

```json
{ "memory": { "snapshot_on_commit": true } }
```

Les instantanés se consultent aussi avec `ov snapshot log --paths
viking://~/memories` et `ov snapshot show`.

Code : `_commit_memory_snapshot` dans `openviking/session/compressor_v3.py`.
