# srgan

La définition d’une image définit le nombre de pixels par pouces et on y réfère pour décrire la qualité de celle-ci puisqu’elle détermine le niveau de détails. De nos jours, les images à haute définition sont essentielles dans différents domaines et applications telles que pour l’imagerie médicale et satellite, le domaine multimédia, la surveillance et la reconnaissance faciale. La super résolution d’une image (SISR) est un processus où l’on prend une image à basse résolution et on produit la même avec plus de détails et avec une plus grande définition. On peut aussi faire référence à ce processus comme un rehaussement définition. Le SISR est un problème complexe, puisque pour obtenir l’image haute définition, il faut prédire les valeurs des pixels à partir de l’image initiale. Pour certains domaines, comme le médical, on ne peut pas générer n’importe quelle information. Pour d’autres domaines, le simple fait que l’image soit plus attrayante pour l’oeil est suffisant.
<br>
Ce projet est une application qui permet de générer une image super définition à l’aide d'un generative adversarial network (GAN). Il est basé sur le travail de Dong et al. (SRCNN - https://arxiv.org/pdf/1501.00092.pdf) et de Ledig et al. (SRGAN - https://arxiv.org/pdf/1609.04802.pdf).
<br><br>

<ul>
Pour lancer un  entrainement, exécutez le script "<i>trainer/main.py</i>". Voici une liste des arguments possibles: <br>
    <b>--data-path</b> (obligatoire): chemin vers les données d'entrainement (les chemins pour gcloud commencent par "<i>gs://</i>")<br>
    <b>--location</b> : [gcloud, <b>local</b>] est-ce que les images sont enregistrées sur <i>google cloud</i> ou
     localement (<b>défaut</b>: gcloud). <br>
    <b>--extension-file</b> :  les types d'extension à considérer (<i>ex: jpg png</i>)<br>
    <b>--ckpnt-gen</b> : où enregistrer les poids du modèle générateur (<i>.h5</i>)<br>
    <b>--ckpnt-discr</b> : où enregistrer les poids du modèle discriminateur (<i>.h5</i>)<br>
    <b>--history-path</b> : où enregistrer l'historique erreurs (<i>.csv</i>)<br>
    <b>--epoch</b> : nombre d'époques (<b>défaut</b>: 30) <br>
    <b>--step</b> : nombre de step par époque (<b>défaut</b>: 100) <br>
    <b>--batch_size</b> : (<b>default</b>: 4)<br>
    <b>--lr-factor</b>: facteur de division pour le sous-échantillonnage à l'entrainement (<b>défaut</b>: 4) <br>
    <b>--weights-gen-path</b> : chemin pour charger les poids d'un modèle générateur préentrainé (doit être un .h5)<br>
    <b>--weights-discr-path</b> : chemin pour charger les poids d'un modèle discriminateur préentrainé (doit être un
     .h5)<br>
     <b>--job-dir</b> : voir https://cloud.google.com/sdk/gcloud/reference/ml-engine/jobs/submit/training
     <br><br>
     <u>exemple</u>: trainer/main.py <b>--data-path</b> path/to/images <b>--location</b> local <b
     >--epoch</b> 200 <b>--step</b> 100 <b>--batch_size</b> 8 <b>--ckpnt-gen</b> ./save/generator/ckpnt_generator.h5
       <b>--ckpnt-disc</b> ./save/discriminator/ckpnt_discriminator.h5
</ul>
<br>
<ul>
Pour analyser les résultats, lancer les scripts "<i>main.py</i>" dans les sous-dossiers de "<i>tester/</i>".
    <ul>
    <b>view_loss_history</b> : sort l'évolution des <i>loss</i> de l'entrainement sous forme de graphiques.
    </ul>
 <br>
    <ul>
    <b>view_model_results</b> : Affiche le résultat et l'analyse d'images passées dans le générateur.<br>
    <b>--gen-path</b> : chemin vers les poids du modèle (doit être un <i>.h5</i>) <br>
    <b>--images</b> : chemin vers l'image à analyser. Peut-être passé sous forme de liste pour analyser plusieurs
     images.<br>
     <b>--factor</b> : facteur par lequel on veut sous-échantillonner l'image de référence.<br>
    </ul>
 <br>
    <ul>
    <u><b>enhance_image</b></u> : Prends une ou des images en entrée et ressors une ou des images rehaussées.<br>
    <b>--gen-path</b> (obligatoire): chemin vers les poids du modèle (doit être un <i>.h5</i>) <br>
    <b>--images</b> (obligatoire): chemin vers la ou les images à rehaussées. Peut-être passé sous forme de liste pour analyser
     plusieurs images.<br>
    <b>--new-size</b> (obligatoire): Peut être un facteur de multiplication des dimensions (ex: 2) ou une nouvelle dimension (ex
     : 800,900). Doivent être des entiers.
    <br>
    <b>--save-to</b> (obligatoire): dossier où les images seront enregistrées sous la forme "<i>enhance_image_name.jpg|png|...</i>"
    <br><br>
    <u>exemple</u>: tester/enhance_image/main.py <b>--gen-path</b> .\save\generator\ckpnt_generator.h5 <b>--image</b
    > path/to/image1.jpg path/to/image2.png <b>--new-size</b> 850,730 <b>--save-to</b> path/to/save/image
    </ul>
</ul>
 <br>
<b>*La branche "<i>model_3</i>" contient un modèle déjà entrainé sur 400 époques. Il se trouve dans "<i>save/generator
</i>".</b>
