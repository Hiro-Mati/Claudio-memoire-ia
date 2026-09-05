# Palier 3 : une infrastructure multi-agents de confiance

Ce palier est celui où le fork atteint ses limites de session unique. Les
trois livrables ci-dessous sont réels, testés et documentés, mais chacun a
un périmètre borné, indiqué explicitement.

## 1. Rôles de processus : api, worker, all

Objectif : que l'ingestion lourde (résumés, embeddings, extraction de mémoire)
ne pèse pas sur la latence de rappel des agents.

`server.queue_role` (ou `openviking-server --role ...`) :

| Rôle | Comportement |
|------|--------------|
| `all` (défaut) | sert HTTP et consomme les files, comme en amont |
| `api` | sert HTTP seulement ; les files (QueueFS, SQLite partagé) sont remplies mais pas consommées |
| `worker` | consomme les files sans ouvrir de port ; plusieurs workers peuvent tourner |

```
openviking-server --role api
openviking-server --role worker        # dans un autre terminal, autant de fois que voulu
```

Le verrou exclusif du workspace (`.openviking.pid`) est levé pour les rôles
séparés, car QueueFS et les verrous de chemin RAGFS coordonnent déjà les
processus.

**Limite vérifiée sur cette machine :** le moteur vectoriel embarqué
(`storage.vectordb.backend = local`) est mono-processus (verrou `store/LOCK`).
Le serveur refuse donc un rôle séparé avec ce backend et l'explique ; les
rôles séparés exigent un backend vectoriel partagé (`volcengine`, VikingDB
hébergé). Un backend vectoriel local multi-processus est le chantier suivant.

Code : `queue_consumers_enabled` et le verrou dans `openviking/service/core.py`,
`_run_worker_process` dans `openviking/server/bootstrap.py`. Tests :
`tests/service/test_queue_role.py`.

## 2. Contrats de contexte

Un agent déclare une fois ce qu'il accepte d'un rappel (budget de tokens,
quotas par type de mémoire, niveau de détail, fenêtre de déduplication, portée
des pairs, réécriture en digest). Le contrat est stocké sous un nom dans les
réglages de l'utilisateur et appliqué à toute recherche `mode="context"` (ou
`/recall`) qui le nomme. Les champs posés explicitement dans la requête
gagnent toujours : un contrat est un jeu de garanties par défaut, pas une
cage.

| Endpoint | Rôle |
|----------|------|
| `GET /api/v1/user-settings/context-contracts` | lister |
| `PUT /api/v1/user-settings/context-contracts/{nom}` | créer ou remplacer (corps : `max_tokens`, `quotas`, `purpose`, `detail`, `dedup_turns`, `peer_scope`, `other_peer_penalty`, `rewrite`, `rewrite_max_bullets`, `query_expansion`, `score_threshold`, `exclude_uris`, `limit`, `description`) |
| `DELETE /api/v1/user-settings/context-contracts/{nom}` | supprimer |
| `POST /api/v1/search/search` avec `"mode": "context", "contract": "nom"` | appliquer ; la réponse indique dans `stats.contract` les champs fournis par le contrat |

```bash
curl -X PUT http://127.0.0.1:1933/api/v1/user-settings/context-contracts/claude-code \
  -H "Content-Type: application/json" \
  -d '{"description": "Claude Code recall", "max_tokens": 1500, "purpose": "coding", "dedup_turns": 5}'
curl -X POST http://127.0.0.1:1933/api/v1/search/search -H "Content-Type: application/json" \
  -d '{"query": "...", "mode": "context", "contract": "claude-code"}'
```

Code : `openviking/retrieve/context_assembler/contracts.py`, `UserConfig.context_contracts`,
`_apply_named_contract` dans `openviking/server/routers/search.py`. Tests :
`tests/retrieve/test_context_contracts.py`.

## 3. Fédération : paquets OVPack signés

Le format OVPack lie déjà chaque fichier par SHA-256 et un condensé global
du manifeste. Le fork ajoute une signature Ed25519 du manifeste, embarquée
dans l'archive (`_ovpack/manifest.sig.json`) avec la clé publique du
signataire, et une politique d'import.

```
python -m openviking.storage.ovpack.signing --generate ~/.openviking/federation.key
```

```json
{
  "federation": {
    "signing_key_file": "~/.openviking/federation.key",
    "key_id": "varua-laptop",
    "trusted_public_keys": ["<hex de la clé publique d'un pair>"],
    "require_signature": false
  }
}
```

- à l'export, si `signing_key_file` est défini, chaque `.ovpack` est signé ;
- à l'import, une signature présente est toujours vérifiée : manifeste altéré
  ou clé inconnue (quand `trusted_public_keys` est renseignée) → refus ;
  `require_signature` refuse en plus les paquets non signés.

Les mémoires restent locales : seuls les sous-arbres exportés circulent. Le
partage lui-même (envoi du fichier, planification) reste manuel ou scripté ;
il n'y a pas encore de protocole de synchronisation entre serveurs.

Code : `openviking/storage/ovpack/signing.py`, `_sign_archive` et
`_verify_archive_signature` dans `openviking/storage/ovpack/operations.py`,
`openviking_cli/utils/config/federation_config.py`. Tests :
`tests/storage/test_ovpack_signing.py`.

## Ce qui reste hors de portée d'une session

- un backend vectoriel local partageable entre processus (prérequis pour
  exploiter les rôles séparés sans service hébergé) ;
- un protocole de fédération (découverte des pairs, synchronisation
  incrémentale, révocation de clés) ;
- des files externes (Redis, NATS) à la place de SQLite pour passer à l'échelle.
