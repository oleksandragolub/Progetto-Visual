# Progetto-Visual

Corso: Visual Information Processing and Management

Anno: 2025/2026

Componenti del gruppo:

— Oleksandra Golub (856706)

— Andrea Spagnolo (879254)

Ambiente utilizzato: Kaggle

Accelerator: GPU P100

Tutti gli esperimenti di training e valutazione dei modelli sono stati eseguiti in ambiente Kaggle con accelerazione GPU.

***

- cartella "codice_finale"

Questa cartella contiene SOLO le versioni definitive del codice. 

Include i file finali con le configurazioni migliori per ogni task:
- Task 1 (ResNet18 con seed + applicato su visual_exam_datataset_2)
- Task 2 (Fine-tuned model + only augmentation + test set degradato)
- Task 3 (SP-HOG + SP-LBP + PCA + Logistic Regression)

File di supporto utilizzati per:
- analisi dei dataset e delle immagini (analisi_datasets.ipynb)
- generazione sintetica dei dati per il bilanciamento delle classi (espansione_sintetica_dati_per_bilanciamento_classi.ipynb)
- evaluation dei modelli salvati (evaluate_any_model.py per Task1/Task2 e evaluate_task3_joblib.py per Tssk3)

***

- cartella "pesi"

Questa cartella contiene tutti i file dei modelli salvati dopo il training finale dei tre task:
- Per Task 1 e Task 2 i modelli sono salvati in formato .pth
- Per Task 3 il modello è stato salvato in formato .joblib (pipeline sklearn completa)

I file corrispondono alle versioni dei modelli che hanno ottenuto le performance migliori.

Questi file permettono di rieseguire evaluation senza rifare il training e riprodurre i risultati riportati nella presentazione finale. 

Nella cartella codice_finale sono presenti gli script:
- evaluate_any_model.py
- evaluate_task3_joblib.py
che possono essere eseguiti da terminale per valutare un modello salvato (.pth o .joblib) su un test set di interesse.

***

- cartelle "task1 / task2 / task3"

Queste cartelle contengono tutto il codice sviluppato durante il progetto, organizzato per task.
Ogni cartella include:
- versioni iniziali (baseline)
- versioni intermedie testate e scartate 
- versioni finali e file di supporto utilizzati durante esperimenti 
- file readme.txt per orientarsi meglio nelle cartelle

***

- cartella "pipeline"

Questa cartella contiene tutti gli schemi delle pipeline utilizzate nei Task 1, Task 2 e Task 3.

Include per ogni task:
- Schema della pipeline baseline
- Schema della pipeline finale con configurazione ottimizzata

***

- file "sports_label.csv" contiene la struttura completa del dataset utilizzato nel progetto.


