# 🦏🎓 RAPPORT TP DEEP LEARNING - INSA Lyon 5IF 🎓🦏
_Paul-Emmanuel SOTIR <paul-emmanuel.sotir@insa-lyon.fr> <paul-emmanuel@outlook.com>_
- Course link https://perso.liris.cnrs.fr/christian.wolf/teaching/deeplearning/tp.html
- Github repository: https://github.com/PaulEmmanuelSotir/BallDetectionAndForecasting
  
### Introduction

...

##### Tache 1 : detection de balles

...

##### Tache 2 : prédiction de la position future de balles

...

### Run instructions

Les dépendences du projets sont gèrées avec un environement conda; Il suffit donc d'une distribution conda (e.g. Anconda ou Miniconda) et de créer l'environement conda pour pouvoir executer le code :  

``` shell
############## Installation ##############

git clone git@github.com:PaulEmmanuelSotir/BallDetectionAndForecasting.git
conda env create -f ./environement.yml
conda activate pytorch_5if
# Télécharge les datasets (nescessite les packages 'curl' et 'tar' sur une distribution Linux (ou WSL - Linux subsystem on Windows))
bash ./download_dataset.sh

############## Exemples d'utilisation ##############

# Entraine le modèle de détection de balles (tache 1) avec les meilleurs hyperparamètres trouvés
python ./src/train.py --detect
# Execute une recherche d'hyperparamètres pour la detection de balles (hyperopt)
python ./src/train.py --detect
# Entraine le modèle de prédiction de position de balles (tache 2) avec les meilleurs hyperparamètres trouvés
python ./src/train.py --pred_seq
# Execute une recherche d'hyperparamètres pour la prédiction de position de la balles (hyperopt)
python ./src/train.py --pred_seq
```

### Developement et docuementation

##### Architecture du projet

Le projet est constitué d'un module Python 'balldetect' contenant 

Organisation du dossier du projet:
``` python
- BallDetectionAndForecasting
  - src/                        # Code source du projet
    - balldetect/               # Module 'balldetect'
      - __init__.py
      - ball_detector.py        # Modèle de detection de balles (tâche 1)
      - datasets.py             # Objets datasets (fourni) et création des dataloaders
      - seq_prediction.py       # Modèle de prédiction de position de balles (tâche 2)
      - torch_utils.py          # Fonctions définissant du code commun aux deux modèles: Couches de convolution (ConvLayer); Couches denses (FCLayer); parralelize (nn.DataParallel), flatten et progress_bar (tqdm personnalisé); 
      - vis.py                  # Visualization des images avec boundings boxes (fourni)
    - train.py                  # Code utilisant le module 'balldetect' pour entrainer le modèle de detection ou de prédiction de position de balles
    - hp.py                     # Code de recherche d'hyperparamètres
  - notebooks/
    - BallDetection.ipynb       # Notebook 'brouillon' pour le test et le developpement de code
    - test_fastai.ipynb         # Notebook 'brouillon' testant une approche préliminaire au TP avec fastai (tache 1.1)
    - test_fastai-bbox.ipynb    # Notebook 'brouillon'testant une approche préliminaire au TP avec fastai (tache 1.2: version avec bounding-boxes)
  - datasets/
    - mini_balls/
    - mini_balls_seq/
  - download_dataset.sh         # Script de téléchargement des datasets
  - environement.yml            # Environement conda, définit les dépendances
  - LICENSE
  - README.md
  - Rapport.md
  - .gitignore
```

##### Petites optimizations du code

- Le(s) GPU disponnibles seront utilisés automatiquement si possible: 

``` python
# Device donné en argument de la fonction '.to(DEVICE)' des torch.Tensor
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
```

- Les dataloaders utilisent plusieur sous-processus en fonction du nombre de coeurs CPU disponnibles sur la machine (paramètre __```num_workers```__ des Dataloader-s) :

``` python
# Nombre de 'thread' utilisés pour chaque dataloader
_DEFAULT_WORKERS = 0 if __debug__ else min(os.cpu_count() - 1, max(1, os.cpu_count() // 4) * max(1, torch.cuda.device_count()))
```

- Pour une meilleur optimization des opérations de Pytorch sur GPU, on peut activer l'auto-tuning (basé sur un benchmark) de la librairie CuDNN s'ajoutant à CUDA. CuDNN est une librairie de NVidia (un fichier cpp et une header C++ ajouté au toolkit CUDA) qui offre des optimisations spécifiques aux réseaux de neurones (Convolutions, produits de matrices, calcul du gradient, ...). CuDNN est intégré à Pytorch comme pour CUDA qui est simplement une dépendance installée dans l'environement Conda. Pytorch permet d'améliorer les performances de CuDNN d'avantage avec les paramètres __```cudnn.benchmark```__ et __```cudnn.fastest```__.
On observe une amélioration de la vitesse d'entrainement entre 30% et 40% pour les modèles Pytorch de detection et de prédiction de position de balles:

``` python
# Torch CuDNN configuration
cudnn.benchmark = torch.cuda.is_available()  # Enable builtin CuDNN auto-tuner
cudnn.fastest = torch.cuda.is_available()  # Disable this if memory issues
```

- Les performances ont aussi été un facteur important dans les choix d'hyperparamètres et d'architecture des modèles (espace de recherche des hyperparametres avec hyperopt). Par exemple, l'architecture du detecteur contient des couches de convolution avec des 'stride' de 2 pour réduire la quantitée de paramètres. Les deux modèles ont un nombre total de couches relativement faibles. Pareillement, le nombre de filtres de convolution, la taille des filtres (3x3 ou, plus haut dans la convolution, 5x5) et la largeur des couches fully-connected ont été choisis, en autre, pour être assez faibles étant donné la simplicité aparante de la tache de detection et pour permettre un entrainement plus rapide.

- Les entrainements des modèles sont également accélérés en parallelisant les 'training steps' sur plusieurs GPUs automatiquement avec __``` model = nn.DataParallel(model) ```__ (voir la fonction __``` paralellize(model: nn.Module) -> nn.Module ```__ dans __```./src/balldetect/torch_utils.py```__). Les modèles ont été entrainé sur une machine personnelle dotée de deux NVidia 1080 ti; La paralèllisation des données avec cette fonction automatique de Pytorch vas donc executer deux entrainements en parallèle et synchroniser les gradients à chaque étapes de manière synchrone en calculant la moyenne des deux 'training steps' avant de passer à l'entrainement sur les prochains batchs.  
De par le besoin de synchronisation des gradients, le modèle et/ou les données/batch_size doivent être assez volumineux pour que cette parallèlisation offre une accélération de l'entrainement par rapport à l'utilisation d'un seul GPU.  
Par exemple, on observe pour le modèle de detection de balles qu'il faut une ```batch_size``` supérieure à 64 pour que les deux 1080 ti soit utilisées au delà de 50% (nvidia-smi). Cependant, trop augmenter la batch_size peut poser des problèmes, notamment à cause de la taille limitée du dataset et, pour des modèles plus importants, pourrais demander une quantitée de mémoire vidéo trop grande.  

##### Tests préliminaires avec fastai (voir notebooks 'brouillons' ```test_fastai.ipynb``` ```test_fastai-bbox.ipynb```)

Dans un premier temps, avant de coder l'approche avec des modèles Pytorch personalisés. Des premiers tests on été fait avec fastai pour avoir une idée de la difficulté de la première tâche et ainsi avoir une baseline. L'approche avec fastai, certes très peut didactique ou optimisée en termes de taille de modèle, permet des résultats assez corrects de manière très rapide. Il s'agissait de tester des modèles connus (e.g. variantes de ResNet) préentrainnés sur des images ImageNet ou autre. Un rapide fine-tunning sur le dataset de détéction de balles permet d'obtenir un detection raisonable en 3-4 lignes de code.  
Une des fonctionalités également interessantes de fastai est l'implémentation d'un méthode pour déterminer un meilleur learning rate sans avoir à executer une recherche d'hyperprametres classique (type random-search): le learning rate est déterminé le temps d'une seule époque en changeant le learning rate à chaque batch d'entrainement (voir [callbacks.lr_finder de fastai](https://docs.fast.ai/callbacks.lr_finder.html)).  
Cette approche préliminaire a permit de mieux connaitre les avantages et inconvénients d'une utilisation très basique de fastai (en effet, fastai n'empêche pas un controle complet sur le modèle entrainé et la procédure d'entrainement). Les résultats obtenus permettent dans la suite de pouvoir mieux interpreter les valeurs des métriques sur ce dataset (donne une baseline raisonnable).

##### Tache 1: Modèle de detection de balles (voir ```./src/balldetect/ball_detector.py``` et les hyperparamètres associés dans ```./src/train.py```)

Le modèle que nous avons construit pour la detection de balles devait dans un premier temps ne détecter que les couleurs de balles et non leur positions.

La loss utilisée pour entrainer le modèle ... TODO: ...

La recherche d'hyperparamètre à permit de trouver de bien meilleurs paramètres d'entrainement et choisir la bonne variante d'architecture parmit celles définies dans l'espace d'hyperparamètres.  
Est donné ci-dessous l'espace de recherche des hyperparamètres donné à hyperopt (algorithme ```tpe.suggest``` avec 200 entrainements de 70 epochs et un early_stopping de 12 epochs):

``` python
# Define hyperparameter search space
hp_space = {
    'optimizer_params': {'lr': hp.uniform('lr', 5e-6, 1e-4), 'betas': (0.9, 0.999), 'eps': 1e-8,
                            'weight_decay': hp.loguniform('weight_decay', math.log(1e-7), math.log(1e-3)), 'amsgrad': False},
    # 'scheduler_params': {'max_lr': 1e-2, 'pct_start': 0.3, 'anneal_strategy': 'cos'},
    'batch_size': hp.choice('batch_size', [32, 64, 128]),
    'architecture': {
        'act_fn': nn.LeakyReLU,
        # TODO: Avoid enabling both dropout and batch normalization at the same time: 1ee ...
        'dropout_prob': hp.choice('dropout_prob', [1., hp.uniform('nonzero_dropout_prob', 0.45, 0.8)]),
        # 'batch_norm': {'eps': 1e-05, 'momentum': 0.1, 'affine': True},
        # Convolutional backbone block hyperparameters
        'conv2d_params': hp.choice('conv2d_params', [
            [{'out_channels': 4, 'kernel_size': (3, 3), 'padding': 1},
                {'out_channels': 4, 'kernel_size': (3, 3), 'padding': 1},
                {'out_channels': 4, 'kernel_size': (3, 3), 'padding': 1, 'stride': 2},
                {'out_channels': 8, 'kernel_size': (5, 5), 'padding': 2},
                {'out_channels': 8, 'kernel_size': (7, 7), 'padding': 3}],

            [{'out_channels': 2, 'kernel_size': (3, 3), 'padding': 1},
                {'out_channels': 4, 'kernel_size': (3, 3), 'padding': 1},
                {'out_channels': 4, 'kernel_size': (3, 3), 'padding': 1, 'stride': 2},
                {'out_channels': 8, 'kernel_size': (5, 5), 'padding': 2},
                {'out_channels': 8, 'kernel_size': (5, 5), 'padding': 2, 'stride': 2},
                {'out_channels': 16, 'kernel_size': (7, 7), 'padding': 3}],

            [{'out_channels': 4, 'kernel_size': (3, 3), 'padding': 1},
                {'out_channels': 4, 'kernel_size': (3, 3), 'padding': 1},
                {'out_channels': 8, 'kernel_size': (3, 3), 'padding': 1},
                {'out_channels': 8, 'kernel_size': (3, 3), 'padding': 1, 'stride': 2},
                {'out_channels': 16, 'kernel_size': (5, 5), 'padding': 2}],

            [{'out_channels': 4, 'kernel_size': (3, 3), 'padding': 1},
                {'out_channels': 4, 'kernel_size': (3, 3), 'padding': 1},
                {'out_channels': 4, 'kernel_size': (3, 3), 'padding': 1},
                {'out_channels': 4, 'kernel_size': (3, 3), 'padding': 1, 'stride': 4},
                {'out_channels': 8, 'kernel_size': (3, 3), 'padding': 1},
                {'out_channels': 8, 'kernel_size': (3, 3), 'padding': 1}]
        ]),
        # Fully connected head block hyperparameters (a final FC inference layer with no dropout nor batchnorm will be added when ball detector model is instantiated)
        'fc_params': hp.choice('fc_params', [[{'out_features': 64}],
                                                [{'out_features': 64}, {'out_features': 128}],
                                                []])}
}
```

La recherche d'hyperparamètres à données les paramètres "optimaux" suivants (best_valid_loss=0.12278, best_train_loss=0.09192 après 67 epcohs):

``` python
{'architecture': {
	'act_fn': nn.LeakyReLU,
	'conv2d_params': ({'kernel_size': (3, 3), 'out_channels': 4, 'padding': 1},
			  {'kernel_size': (3, 3), 'out_channels': 4, 'padding': 1},
			  {'kernel_size': (3, 3), 'out_channels': 8, 'padding': 1},
			  {'kernel_size': (3, 3), 'out_channels': 8, 'padding': 1, 'stride': 2},
			  {'kernel_size': (5, 5), 'out_channels': 16, 'padding': 2}),
	'dropout_prob': 0.7187055796612525,
	'fc_params': ()},
	'batch_size': 32,
	'optimizer_params': {'amsgrad': False, 'betas': (0.9, 0.999), 'eps': 1e-08, 'lr': 9.961462262405672e-05, 'weight_decay': 0.0002119238018958741}}
```

(voir la section __"Résultats/Resultats tache 1"__ ci-dessous pour plus de détails sur les detection faites par ce modèle)

##### Tache 2: Modèle de prédiction de position de balles (voir ```./src/balldetect/seq_prediction.py``` et les hyperparamètres associés dans ```./src/train.py```)

Le modèle de prédiction de position de balles est un simple réseaux de neurones dense (fully connected layers) puisque qu'il n'y a pas d'images en entrée du modèle.  

La procédure d'entrainement du modèle est relativement similaire à celle du modèle de detection de la tache 1, aux hyperparamètres près. Nous n'avons malheureusement pas eu le temps d'executer une recherche d'hyperparamètres pour ce modèle. De même, par manque de temps, nous n'avons pas pû travailler autant que voulut sur l'amélioration de ce modèle et l'interprètation de la qualité des prédictions faites sur les positions des balles au delà de la métrique utilisée (voir la section __"Résultats/Resultats tache 2"__ ci-dessous)

### Résultats

##### Resultats tache 1 : Detection des balles

Voici un tableau récapitulant les résultats obtenus avec nos modèles: Modèle préliminaire fastai; Modèle inférant seulement la présence de couleurs de balles; Modèle final inférant les couleurs et position des balles dans l'image d'entrée.

![task1_results]()

##### Resultats tache 2 : Prédiction de la position de balles

Voici un tableau récapitulant les résultats obtenus avec notre modèle de prédiction de la position future de la balle à partir de sa position précédente :

![task2_results]()

### Conclusion

Pour conclure, développer et tester des modèles Pytorch, certes simples, m'as permit d'approfondir mes connaissances notamment d'un point de vue pratique/technique.  

Il est regrettable que la qualité du modèle de la tâche 2 souffre probablement d'un manque de temps (ou plutôt d'une meilleure mise en prioté des différents aspects du projet), cepandant, nous avons pû développer des modèles Pytorch et des procédures d'entrainement assez complet, générique et réutilisable. En effet, ce projet a également été l'occasion poser une base de code relativement solide pour de futurs projets.

##### Autres pistes et améliorations possibles

+ redondance entre couleurs et bounding boxes
+ recherche d'hyperparamètres pour la prédiction de position future des balles
+ méthodes de recherche d'hyperparamètres plus efficaces (e.g. la méthode utilisée par fastai dans [callbacks.lr_finder](https://docs.fast.ai/callbacks.lr_finder.html): [post de blog de Sylvain Gugger](https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html)) et utilisation de [microsoft/nni](https://github.com/microsoft/nni) regroupant de nombreuses de méthodes de recherche d'hyperparamètres)
+ utiliser des méthodes de recherche d'architecture automatiques (beaucoup d'engouement/progrès dans la communauté deeplearning autour des méthodes de "neural net architecture search" et "meta-learning")
+ tests plus poussés du scheduling de learning rate (e.g. investiger pourquoi OneCycle learning rate scheduler n'a pas donné de résultats probants sur la détection de balles avec notre modèl)
+ tests plus poussés avec de la batch norm: implémentée mais très peu testée pour réduire l'espace de recherche d'hyperparamètres de par le manque de temps (mais dropout, weight decay, .. utilisé)
+ Utiliser de l'augmentation de données aurait pû être intéressant

![Logo INSA Lyon](https://www.insa-lyon.fr/sites/www.insa-lyon.fr/files/logo-coul.jpg =200x))
_Copyright (c) 2019 Paul-Emmanuel SOTIR_  
**_This project and document is under open-source MIT license, browse to: https://github.com/PaulEmmanuelSotir/BallDetectionAndForecasting/blob/master/LICENSE for full MIT license text._**
