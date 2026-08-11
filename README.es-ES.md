

# [ECAI 2024] MakeupAttack

Esta es la implementación oficial de nuestro artículo ["MakeupAttack: Feature Space Black-box Backdoor Attack on Face Recognition via Makeup Transfer"](), ECAI2024.

## Cita

Por favor, cite nuestro artículo en su publicación si le resulta útil para su investigación:

```latex
@article{sun2024makeupattack,
  title={MakeupAttack: Feature Space Black-box Backdoor Attack on Face Recognition via Makeup Transfer},
  author={Sun, Ming and Jing, Lihua and Zhu, Zixuan and Wang, Rui},
  journal={arXiv preprint arXiv:2408.12312},
  year={2024}
}
```

## Flujo de Trabajo Principal

## ![figure2](./img/figure2.jpg)Configuración

### Entornos

Este proyecto se desarrolló con Python 3.7 en Ubuntu 18.04. Ejecute el siguiente script para instalar los paquetes requeridos:

```shell
pip install -r requirements.txt
```

### Conjuntos de Datos

El MT-Dataset publicado, utilizado para el entrenamiento del generador, se puede descargar desde [Google Drive](https://drive.google.com/drive/folders/1PjwsEGachV1Pqy1nlp74eZ9MeNDGzWCA?usp=sharing).

El subconjunto benigno y envenenado de **PubFig** publicado se puede descargar desde [Google Drive](https://drive.google.com/drive/folders/1P2kJgoRmtITN3POX_C_CstVP8pkz07kp?usp=sharing). También puede utilizar el código proporcionado para entrenar el generador de activadores y crear sus propios conjuntos de datos envenenados. 

## Entrenamiento Directo

```shell
python train_model.py --phase 'poison' --dataset 'pubfig' --model 'facenet' --makeupdir 'assets/pubfig-makeup'
```

## Entrenamiento Desde Cero

### Preentrenamiento del Generador

```shell
python train_GAN.py
```

### Generación de Muestras Envenenadas

```shell
python generate.py --g_path './assets/GAN/G.pth'
```

### Entrenamiento con Puerta Trasera (Resultado Intermedio)

```shell
python train_model.py --phase 'poison' --dataset 'pubfig' --model 'facenet' --makeupdir 'assets/pubfig-makeup'
```

**Nota**: El modelo entrenado se almacena en la carpeta `ckpt` como `DATASET_NETWORK. pt`(_p. ej._, `pubfig_resnet.pt`).

### Ajuste Fino del Generador

```shell
python train_GAN.py --adv --dataset 'pubfig' --model 'facenet' --model_path './ckpt/model/pubfig_facenet_makeup.pt' --GAN_path './ckpt/GAN'
```

**Nota**: `GAN_path` es una carpeta que contiene `G.pth`, `D_A.pth`, `D_B.pth`, `H.pth`, los cuales pueden copiarse desde la carpeta `log` y renombrarse en consecuencia.

### Entrenamiento con Puerta Trasera

```shell
python train_model.py --phase 'poison' --dataset 'pubfig' --model 'facenet' --transfer --model_path './ckpt/model/pubfig_facenet_makeup.pt' --makeupdir 'assets/pubfig-makeup'
```
**Nota**: El modelo cargado corresponde al resultado del período intermedio.

## Agradecimientos

Construimos nuestro código basándonos en [BackdoorVault](https://github.com/Gwinhen/BackdoorVault) y [AMT-GAN](https://github.com/CGCL-codes/AMT-GAN). ¡Gracias por su excelente trabajo!
