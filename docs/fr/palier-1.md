# Palier 1 : un produit local qui tient sur un portable

Objectif : découpler la qualité de la mémoire de la puissance de la machine.
Chaque section décrit une fonctionnalité du fork Claudio-memoire-ia, sa
configuration et la façon de la vérifier.

## 1. Rafraîchissement des parents regroupé ou paresseux

En amont, chaque dossier dont le résumé vient d'être généré déclenche
immédiatement le rafraîchissement de son dossier parent, puis du parent du
parent, jusqu'à la racine. Sur un arbre profond, chaque ancêtre est donc
résumé une fois par enfant, ce qui multiplie les appels au VLM.

Trois modes sont disponibles dans la section `semantic` de `ov.conf` :

| Mode | Comportement |
|------|--------------|
| `eager` (défaut, identique à l'amont) | le parent est mis en file tout de suite |
| `debounced` | les demandes pour un même parent sont fusionnées et mises en file une seule fois après `parent_refresh_debounce_s` secondes sans nouvel enfant terminé |
| `lazy` | le parent est seulement marqué en attente (compteur `pending_child_changes` de ses fichiers annexes) ; le rafraîchissement est mis en file la première fois que son résumé ou sa vue d'ensemble est lu |

```json
{
  "semantic": {
    "parent_refresh_mode": "debounced",
    "parent_refresh_debounce_s": 60
  }
}
```

Le regroupement est en mémoire du processus : une fenêtre perdue lors d'un
arrêt n'est pas un problème de cohérence, le compteur d'attente reste écrit
dans les fichiers annexes du parent et le prochain changement (ou la prochaine
lecture en mode paresseux) le replanifie.

Code : `openviking/storage/queuefs/semantic_ops/parent_refresh_scheduler.py`,
`SemanticProcessor._enqueue_parent_refresh`, et le déclencheur de lecture
`_SemanticMixin._maybe_schedule_lazy_parent_refresh`.

Tests : `tests/storage/test_parent_refresh_scheduler.py`.

## 2. Résumés par étages

La génération sémantique appelle le VLM deux fois par niveau : une fois par
fichier (résumé court qui alimente la vue d'ensemble) et une fois par dossier
(vue d'ensemble L1, dont on extrait L0). Le résumé d'un fichier ne demande pas
un gros modèle.

La section optionnelle `file_summarizer` de `ov.conf`, de même forme que
`vlm`, désigne le modèle utilisé pour les résumés de fichiers texte. Les vues
d'ensemble de dossiers, l'extraction de mémoire et les médias restent sur
`vlm`. Sans cette section, tout passe par `vlm` comme en amont.

```json
{
  "file_summarizer": {
    "provider": "litellm",
    "model": "ollama/qwen3.5:0.8b",
    "api_key": "no-key",
    "api_base": "http://127.0.0.1:11434",
    "temperature": 0.0,
    "extra_request_body": {"num_ctx": 8192, "think": false}
  }
}
```

`openviking-server doctor` affiche le modèle retenu. Code :
`OpenVikingConfig.get_file_summarizer()` et
`SemanticProcessor._generate_text_summary`. Tests :
`tests/storage/test_file_summarizer_tiering.py`.

## 3. Repli lexical local (BM25) fusionné avec la recherche dense

Un petit modèle d'embedding rate les jetons exacts : noms de fonctions,
identifiants, codes d'erreur. La collection vectorielle locale n'a pas de
recherche par mots-clés (elle lève `NotImplementedError`) et les vecteurs
épars ne sont fournis que par quelques services hébergés.

Le fork ajoute un index BM25 (SQLite FTS5, intégré à Python) dans
`{storage.workspace}/_system/lexical/lexical.db`, alimenté aux écritures de
l'index vectoriel et reconstruit tout seul depuis celui-ci s'il est vide.
Les identifiants sont indexés tels quels et découpés (`parse_abstract_overview`
et `parse abstract overview`), donc le symbole exact et ses mots trouvent.

Fusion dans `find` et `search`, à la recherche globale comme à la descente par
dossier : `score = min(1, dense + lexical_boost × bm25)`. Un résultat exact
absent des candidats denses est récupéré depuis l'index vectoriel avec le score
`lexical_boost × bm25` ; une entrée lexicale dont l'enregistrement vectoriel a
disparu est purgée à la lecture. L'index vectoriel reste la source de vérité.

```json
{
  "retrieval": {
    "lexical_index_enabled": true,
    "lexical_boost": 0.3,
    "lexical_limit": 20
  }
}
```

Code : `openviking/retrieve/lexical_index.py`, miroir dans
`VikingVectorIndexBackend` (`_lexical_mirror`), fusion dans
`HierarchicalRetriever` (`_lexical_hits`, `_apply_lexical_hits`,
`_rebuild_lexical_index`). Tests : `tests/retrieve/test_lexical_index.py`,
`tests/retrieve/test_lexical_fusion.py`.
