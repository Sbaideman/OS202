### Version scalaire : Parallélisation par décomposition de domaine
Pour la version scalaire, nous avons implémenté une stratégie de **décomposition de domaine** par bandes horizontales. L'introduction de **cellules fantômes (ghost cells)** et l'utilisation de `MPI_Sendrecv` pour l'échange de halo à chaque itération ont permis de résoudre les dépendances de données en mémoire distribuée. De plus, nous avons remplacé l'affichage pixel par pixel par un rassemblement global via `Gather` et un rendu par bloc avec `surfarray`, augmentant ainsi considérablement l'efficacité et la fluidité sur de grandes grilles.

---

### Version vectorisée : Synergie entre convolution et MPI
Dans la version vectorisée, nous avons intégré l'algorithme de **convolution 2D** au sein du framework MPI. Chaque processus exploite la puissance du calcul matriciel local pour traiter son sous-domaine, tout en gérant manuellement le repliement horizontal et en s'appuyant sur MPI pour la topologie torique verticale. Cette approche combine parfaitement les performances locales de NumPy avec la scalabilité de l'architecture à mémoire distribuée, constituant une solution optimale pour les simulations à grande échelle.

---

### Instructions d'utilisation
```bash
# Version scalaire
mpiexec -np 1 python game_of_life_parall.py
mpiexec -np 2 python game_of_life_parall.py

# Version vectorisée
mpiexec -np 1 python game_of_life_vect_parall.py
mpiexec -np 2 python game_of_life_vect_parall.py
```

**Remarque :** Lors de l’expérience, nous avons constaté que lorsque le nombre de processus atteignait 4 ou plus, la fenêtre Pygame se figeait et ne répondait plus. Nous supposons que cela est dû à un conflit entre le thread de l’interface graphique et la communication parallèle.