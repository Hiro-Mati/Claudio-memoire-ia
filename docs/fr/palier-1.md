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
