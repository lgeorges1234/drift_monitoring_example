# Jeu de Données de Partage de Vélos - Analyse de Surveillance de Dérive

## Aperçu du Projet

Ce projet implémente un système complet de surveillance de modèles d'apprentissage automatique pour le jeu de données UCI Bike Sharing. L'analyse se concentre sur la détection et la compréhension de la dérive des données entre janvier 2011 (période d'entraînement) et février 2011 (période de production), en comparant spécifiquement les modèles de dégradation des performances hebdomadaires.

### Détails d'Implémentation Technique

- **Modèle** : RandomForestRegressor (50 estimateurs, random_state=0)
- **Caractéristiques** : 7 numériques + 3 catégorielles
- **Évaluation** : RegressionPreset, TargetDriftPreset, DataDriftPreset
- **Tests Statistiques** : Kolmogorov-Smirnov, Test Z
- **Seuil de Dérive** : 0.1 (standard industriel typique)

### Structure du Projet
```
bike_sharing_monitoring/
├── bike_sharing_monitoring_model_validation/     # Validation initiale du modèle
├── bike_sharing_monitoring_production_model/     # Performance du modèle de production
├── bike_sharing_monitoring_weekly_monitoring/    # Rapports de dérive hebdomadaires
├── bike_sharing_monitoring_target_analysis/      # Analyse de dérive de la cible
└── bike_sharing_monitoring_data_drift/          # Analyse de dérive au niveau des caractéristiques
```

## Exécution

### Commande Unique pour Exécuter le Script
```bash
python bike_sharing_drift_monitoring.py
```

### Prérequis
Assurez-vous d'avoir installé toutes les dépendances :
```bash
pip install -r requirements.txt
```
### Visualisation des Résultats
#### Lancer le serveur Evidently UI :
```bash
bashevidently ui --workspace ./datascientest-workspace/
```

#### Accéder aux rapports :
Ouvrir le navigateur à l'adresse : http://localhost:8000
Explorer les 5 projets créés avec leurs rapports respectifs

## Résultats de l'Analyse

### Étape 4 : Analyse des Performances Hebdomadaires (Semaines 1, 2 et 3)

**Ce qui a changé au cours des semaines 1, 2 et 3 :**

L'analyse des performances hebdomadaires a révélé un modèle clair de dégradation du modèle au cours des trois semaines de février :

- **Semaine 1 (29 jan - 7 fév)** : Déclin initial des performances lorsque le modèle a rencontré les premières conditions de février
- **Semaine 2 (7 fév - 14 fév)** : Dégradation continue avec des valeurs RMSE croissantes
- **Semaine 3 (15 fév - 21 fév)** : Détérioration des performances la plus significative, identifiée comme la semaine la moins performante

Les rapports de régression ont montré des augmentations constantes des métriques d'erreur (RMSE, MAE) et des diminutions de la précision du modèle (R²) sur les trois semaines, indiquant un modèle de dégradation progressive plutôt que des chutes soudaines de performance. Cela suggère un changement environnemental continu affectant la capacité prédictive du modèle tout au long de février.

### Étape 5 : Analyse des Causes Profondes - Investigation de la Dérive de la Cible

**Cause profonde de la dérive (basée sur l'analyse des données) :**

L'analyse de dérive de la cible utilisant `TargetDriftPreset` sur la semaine la moins performante (Semaine 3) a révélé :

- **Score de dérive de la cible : 0.063** (en dessous du seuil de signification typique de 0.1)
- **Aucune dérive significative de la cible détectée** utilisant le test de Kolmogorov-Smirnov
- **Dérive des prédictions : Non détectée** (p-value : 0.063368)

Si la dérive de cible était détectée, on verrait :

❌ Changement radical des heures de pointe (ex: 10h-16h au lieu de 8h-18h)\
❌ Changement des patterns weekend vs semaine\
❌ Augmentation/diminution massive de la demande globale\
❌ Nouveaux pics d'usage à des moments inattendus\

Mais nos résultats montrent :
✅ Les heures de pointe restent 8h et 18h\
✅ Les patterns weekend vs semaine sont similaires\
✅ La demande globale est stable\
✅ Pas de nouveaux comportements d'usage\

**Conclusion** : L'absence de dérive significative de la cible indique qu'il s'agit d'un cas de **décalage de covariables** plutôt que de dérive conceptuelle. La relation fondamentale entre les caractéristiques d'entrée et la demande de location de vélos est restée cohérente.

Les utilisateurs n'ont donc **PAS changé leurs habitudes**:

- Ils vont toujours au travail aux mêmes heures
- Ils utilisent encore les vélos pour les mêmes raisons
- Les patterns d'usage restent identiques

Cette découverte élimine la possibilité que des facteurs externes (vacances, changements de politique ou changements comportementaux) aient fondamentalement modifié la façon dont les gens utilisent les services de partage de vélos.

### Étape 6 : Analyse de Dérive des Caractéristiques d'Entrée

**Confirmation de la cause environnementale (basée sur l'analyse des caractéristiques) :**

L'analyse de dérive des données utilisant `DataDriftPreset` sur les caractéristiques numériques de la semaine 3 a révélé :

- **66,7% des caractéristiques en dérive** (6 sur 9 présentent une dérive significative)
- **Dérive météorologique extrême** avec des scores de 0.000000 pour toutes les variables météo
- **Stabilité comportementale parfaite** avec des scores de 1.000000 pour les variables temporelles

#### Si les caractéristiques d'entrée étaient stables, on verrait :
❌ Scores de dérive proches de 1.0 pour toutes les variables\
❌ Distributions météorologiques similaires entre janvier et février\
❌ Pas de changement saisonnier détectable\
❌ Performance du modèle maintenue\

#### Mais nos résultats montrent :
✅ Dérive extrême des températures (temp, atemp : 0.000000)\
✅ Dérive extrême de l'humidité (hum : 0.000000)\
✅ Dérive extrême de la vitesse du vent (windspeed : 0.000000)\
✅ Stabilité parfaite des patterns temporels (hr, weekday : 1.000000)\

**Conclusion** : 

Cette analyse confirme définitivement que **l'ENVIRONNEMENT a changé** :

- **La météo de février est différente de janvier** : transition hivernale → printanière avec des conditions complètement nouvelles
- **Le modèle ne sait pas comment prédire la demande avec cette nouvelle météo** : entraîné sur des données froides d'hiver, il ne peut pas gérer les conditions plus douces de février
- **Même comportement + nouvelle météo = prédictions incorrectes** : les utilisateurs gardent les mêmes habitudes, mais dans un contexte météorologique que le modèle n'a jamais vu

Cette découverte valide notre diagnostic de décalage de covariables et élimine définitivement la possibilité d'une dérive conceptuelle.

## Stratégies d'Adaptation

**Stratégie Recommandée : Adaptation de Modèle Saisonnière**

### Actions Immédiates :
- **Réentraîner le modèle** avec des données couvrant diverses conditions météorologiques
- **Prioriser les relations météo-demande** (toutes les variables météo montrent une dérive extrême)
- **Maintenir les caractéristiques comportementales** (elles restent stables et prédictives)

### Stratégie à Long Terme :
- **Versioning de modèles saisonniers** pour gérer les transitions récurrentes
- **Déclencheurs de réentraînement basés sur la météo** avec seuils de dérive
- **Pipeline d'apprentissage adaptatif** incorporant automatiquement les nouveaux patterns saisonniers

### Surveillance Optimisée :
- **Focus sur les caractéristiques météorologiques** comme indicateurs primaires
- **Vérifications de stabilité comportementale** pour détecter la vraie dérive conceptuelle
- **Système d'alerte précoce** pour les transitions saisonnières

## Conclusion

Cette analyse démontre l'importance de la surveillance de dérive dans les systèmes ML en production. La dégradation des performances est due aux variations saisonnières météorologiques, non aux changements comportementaux.

**Framework reproductible développé :**
- Détecter et diagnostiquer différents types de dérive
- Distinguer décalage de covariables vs dérive conceptuelle  
- Stratégies d'adaptation ciblées basées sur les données
- Surveillance robuste pour environnements de production

**Insights clés :** Les modèles de partage de vélos nécessitent une adaptation saisonnière, les caractéristiques météorologiques sont des indicateurs critiques de dérive, et les comportements utilisateurs restent remarquablement stables.