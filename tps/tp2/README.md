# TD2
#### Tianze Xia

## 1. Parallélisation ensemble de Mandelbrot


### 1.1 Analyse du Speedup pour le Calcul Parallèle de l'Ensemble de Mandelbrot
```bash
# Exécution
mpiexec -np 1 python mandelbrot_mpi.py
...
```

| Nombre de Processus | Temps (s) | Speedup |
|:---------:|:-----------------:|:---------:|
|        1 |             2.18  |  1.00 |
|        2 |             1.11 |  1.96 |
|        4 |             0.63 |  3.46 |
|        8 |             0.51 |  4.27 |
|        16 |            0.42 |  5.19 |

**Accélération des performances :** Avec 2 à 4 processus, l’accélération est quasi linéaire (efficacité d’environ 86.5 % à 98 %). Cependant, lorsque le nombre de processus passe à 8 et 16, l’accélération chute respectivement à seulement 4,27 et 5,19, l’efficacité dégringolant à environ 32 %.

**Déséquilibre de charge (cause principale) :** La charge de calcul de l’ensemble de Mandelbrot est extrêmement inégalement répartie spatialement. La zone noire au centre de l’image (correspondant à l’intérieur de l’ensemble) nécessite l’exécution de toutes les itérations, tandis que les zones périphériques se déséquilibrent rapidement. L’utilisation d’un simple partitionnement par « blocs de lignes contiguës » entraîne une surcharge du processus responsable des lignes centrales, tandis que les processus responsables des lignes du haut et du bas restent inactifs dès le début.

**Effet de barillet :** Conformément au principe du calcul parallèle, le temps d’exécution total dépend du processus le plus lent. Comme le partitionnement statique ne peut pas détecter les différences de densité de calcul, il entraîne un déséquilibre de charge important, limitant ainsi les améliorations de performances en cas de parallélisme élevé.


### 1.2 Analyse des Performances pour la Répartition Statique Optimisée
```bash
# Exécution
mpiexec -np 1 python mandelbrot_cyclic.py
...
```

| Nombre de Processus | Temps (s) | Speedup |
|:------------------:|:--------------:|:---------:|
|                     1 |              2.14 |   1.00 |
|                     2 |              1.06 |   2.02 |
|                     4 |              0.59 |   3.63 |
|                     8 |              0.44 |   4.86 |
|                     16 |             0.35 |   6.11 |

Nous avons mis en œuvre une répartition cyclique (ou entrelacée) des lignes. Au lieu de blocs contigus, chaque processus $i$ calcule les lignes $y$ telles que $y \equiv i(mod\ nbp)$.

**Comparaison :**
Les résultats montrent une nette amélioration du speedup, passant de 5.19 à 6.11 pour 16 processus. Cette stratégie permet de mieux distribuer les zones de calcul intense (le centre de l'ensemble) entre toutes les tâches, réduisant ainsi le déséquilibre de charge.

**Problèmes potentiels :**
Cette approche reste statique. Elle ne s'adapte pas à la complexité locale si l'on effectue un zoom sur une zone spécifique où la charge n'est plus distribuée de manière régulière. De plus, elle peut réduire la localité spatiale des données, ce qui pourrait impacter les performances sur des algorithmes nécessitant des communications entre lignes voisines. Une approche Maître-Esclave (dynamique) est nécessaire pour une robustesse totale.

### 1.3 Analyse des Performances pour la Stratégie Maître-Esclave
```bash
# Exécution (Nécessite nbp >= 2)
mpiexec -np 2 python mandelbrot_cyclic.py
...
```

| Nombre de Processus | Temps (s) | Speedup |
|:--------------------:|:----------------:|:--------:|
|                     2 |              2.05 |   1.00 |
|                     4 |              0.68 |   3.01 |
|                     8 |              0.42 |   4.88 |
|                     16 |             0.31 |   6.61 |

Nous avons implémenté une stratégie Maître-Esclave (Master-Slave). Le processus 0 (Maître) distribue dynamiquement les lignes aux autres processus (Esclaves) au fur et à mesure de leur disponibilité.

**Analyse et Comparaison :**
Pour 16 processus (soit 15 esclaves actifs), le temps descend à 0,31s, ce qui est le meilleur résultat parmi toutes les méthodes testées.
* **Bloc (1.1) :** 0,42s (Déséquilibre fort).
* **Cyclique (1.2) :** 0,35s (Amélioré mais limité).
* **Maître-Esclave (1.3) :** 0,31s (Équilibrage dynamique optimal).
  
**Conclusion :**
L'approche Maître-Esclave est la plus robuste pour l'ensemble de Mandelbrot car elle s'adapte à l'hétérogénéité du calcul. Bien que le Maître soit "perdu" pour le calcul, le gain obtenu par l'équilibrage de charge dynamique compense largement ce sacrifice, surtout avec un grand nombre de processus. Elle garantit qu'aucun processus ne reste inactif tant qu'il reste des lignes à calculer.




## 2. Produit Matrice-Vecteur

### (a) Méthode par Colonnes
```bash
# Exécution
mpiexec -np 1 python matvec_col.py
...
```

### (a) Méthode par Colonnes
| Nombre de Processus | Nloc | Temps (s) | Speedup |
|:----------------:|:------:|:--------:|:-----:|
|         1 |      16384 |     0.048 |  1.00    |
|         2 |      8192 |     0.041 |  1.17    |
|         4 |      4096 |     0.042 |  1.14    |
|         8 |      2048 |     0.025 |  1.92    |
|        16 |      1024 |     0.011 |  4.36    |

### (b) Méthode par Lignes
| Nombre de Processus | Nloc | Temps (s) | Speedup |
|:----------------:|:------:|:--------:|:-----:|
|         1 |      16384 |     0.047 |  1.00    |
|         2 |      8192 |     0.041 |  1.15    |
|         4 |      4096 |     0.039 |  1.21    |
|         8 |      2048 |     0.038 |  1.24    |
|        16 |      1024 |     0.026 |  1.81    |

### Analyse et Conclusion
**Comparaison d'efficacité :** À l'échelle de cette expérience, le partitionnement par colonnes (a) surpasse le partitionnement par lignes (b) en cas de forte concurrence. Ceci s'explique principalement par le fait que le partitionnement par colonnes exploite mieux le cache lorsque la tâche est suffisamment petite, tandis que le partitionnement par lignes est limité par la copie globale du vecteur u et le processus de collecte des résultats.

**Limitation de la mémoire :** La vitesse de traitement du processeur est bien supérieure à la vitesse à laquelle la mémoire fournit les données. Les produits matrice-vecteur nécessitent la lecture de grandes quantités de données, mais la logique de calcul est simple, ce qui entraîne souvent une inactivité du processeur pendant l'attente des données, et par conséquent un gain de vitesse bien inférieur à la linéarité.





## 3. Entraînement pour l'examen écrit

### 3.1 Accélération maximale

Alice a observé que **90% du temps** de son programme peut être parallélisé. Nous utilisons la **loi d'Amdahl**, qui exprime le **speedup maximal** en fonction de la proportion parallélisable **p = 0.9** :

$$
S(n) = \frac{1}{(1 - p) + \frac{p}{n}}
$$

En prenant la limite lorsque **n → ∞** (nombre de nœuds de calcul très élevé), la formule devient :

$$
S_{\max} = \frac{1}{1 - 0.9} = \frac{1}{0.1} = 10
$$

Donc, **même avec un nombre infini de nœuds**, Alice ne pourra jamais dépasser une accélération de **10x**.

---

### 3.2 Nombre optimal de nœuds

Dans un contexte réel, il est important de choisir un nombre de nœuds suffisant **sans gaspiller de ressources CPU**. Selon la loi d'Amdahl :

- Avec **n = 4** nœuds :

  $$
  S(4) = \frac{1}{(1 - 0.9) + \frac{0.9}{4}}
  $$

  $$
  S(4) = \frac{1}{0.1 + 0.225} = \frac{1}{0.325} \approx 3.08
  $$

Cela montre que **l'ajout de plus de 4 nœuds ne serait pas très bénéfique**. Un nombre raisonnable de nœuds se situerait entre **4 et 8**, car au-delà, le gain est marginal.

---

### 3.3 Accélération selon la loi de Gustafson

Lorsque **la quantité de données est doublée**, nous supposons une complexité parallèle **linéaire** et utilisons la **loi de Gustafson** :

$$
S(n) = n - (1 - p) \times n
$$

Sachant qu'Alice a obtenu une accélération **S(n) = 4** pour un certain nombre de nœuds **n**, nous cherchons la nouvelle accélération **S'(n)** en doublant la charge de travail.

Avec **p = 0.9**, et en supposant que le même nombre de nœuds **n** est utilisé :

$$
S'(n) = n - (1 - 0.9) \times n
$$

$$
S'(n) = n - 0.1 \times n = 0.9n
$$

Cela signifie que **l'accélération augmentera presque linéairement avec la charge de travail**. Si **S(n) = 4** auparavant, alors avec **2× plus de données**, Alice pourrait atteindre :

$$
S'(n) = 2 \times 4 = 8
$$

Donc, en doublant les données, Alice pourrait espérer une accélération **jusqu'à 8x** en utilisant la loi de Gustafson.
