# TD3
#### Tianze Xia

## 1. Parallélisation du Bucket Sort
```bash
# Exécution
mpiexec -np 1 python bucket_sort.py
...
```

| Nombre de Processus | Temps (s) | Speedup |
|:---------:|:-----------------:|:---------:|
|        1 |             6.47  |  1.00 |
|        2 |             6.02 |  1.07 |
|        4 |             4.51 |  1.43 |
|        8 |             3.66 |  1.77 |
|        16 |            3.85 |  1.68 |

**Analyse des résultats :**
* **Gain limité :** Le speedup maximal atteint n'est que de 1,77 avec 8 processus, ce qui est loin de l'accélération linéaire idéale. Cela indique que l'algorithme est fortement limité par le surcoût des communications (notamment les opérations Scatter, Gather et surtout le All-to-all pour la redistribution des données).
  
* **Point de saturation :** On observe une dégradation des performances à 16 processus (3,85s contre 3,66s à 8 processus). À ce stade, le coût lié à la gestion d'un plus grand nombre de messages et à la synchronisation l'emporte sur le gain de calcul local.

**Conclusion :** Pour cette taille de problème, le ratio calcul/communication n'est pas optimal pour une parallélisation massive. Le seuil de rentabilité se situe à 8 processus dans cet environnement.