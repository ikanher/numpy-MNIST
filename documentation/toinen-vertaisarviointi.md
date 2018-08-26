# Toinen vertaisarvioininti

Tehtävän oli tehdä tehdä vertailu toisen pelkkää Numpy:tä käyttävän neuroverkkokirjaston kanssa.

Vertailu siis suoritetteen [Neurose](https://github.com/irenenikk/neurose) kirjaston kanssa.

## Ensimmäiset askeleet

Kokeilin suorittaa `mnist_example.py` ohjelman. Ensin ohjelma latasi Pytorchia käyttäen MNIST kirjaston. Tämän jälkeen ei tapahtunutkaan mitään. Kotvan odoteltuani, päätin lopettaa ohjelman Ctrl-C komennolla.

Lisäsin muutaman debug printin ohjelmaan ja huomasin, että kyllä se tekee jotain, mutta kestää vain todella kauan. Jäin odottelemaan. Selvisi, että yhden epochin treenaamiseen kuluu pöytäkoneeni i5-7600K prosessorilla yli kaksi minuuttia.

### Mikä maksaa?

Hankalaksihan tämä raportin tekeminen menee, jos heti alkuun joudun odottamaan pari kolme tuntia, jotta verkko olisi vakioasetuksilla (100 epochia) treenattu.

Testasin seuraavaksi tuomalla mallin omaan kirjastooni. Tämä tapahtui hyvin yksinkertaisesti. Editoimalla `models.py` tiedostoa ja luomalla sinne uuden mallin.

```
class NeuroseModel():
    """
    The same model that is used in

    https://github.com/irenenikk/neurose
    """

    def __init__(self):
        """
        Construct a new NeuoreModel

        n_inputs - number of inputs
        n_hidden1 - number of nodes in the first hidden layer
        n_hidden2 - number of nodes in the second hidden layer
        n_output - number of outputs
        """
        # model parameters
        self.n_input = 28*28
        self.n_hidden1 = 256
        self.n_hidden2 = 120
        self.n_hidden3 = 64
        self.n_output = 10

    def get_layers(self):
        """
        Returns a list of layers that make this model
        """
        layers = []
        layers.append(ReLU())                                   # activation
        layers.append(Linear(self.n_input, self.n_hidden1))     # input layers
        layers.append(ReLU())                                   # activation
        layers.append(Linear(self.n_hidden1, self.n_hidden2))   # first hidden layer
        layers.append(ReLU())                                   # activation
        layers.append(Linear(self.n_hidden2, self.n_hidden3))   # second hidden layer
        layers.append(ReLU())                                   # activation
        layers.append(Linear(self.n_hidden3, self.n_output))    # second hidden layer
        layers.append(Softmax())                                # output layer
        return layers

    def get_loss_func(self):
        """
        Returns the loss function to be used
        """
        return CrossEntropy()
```

Nyt malli voidaan importata ja ottaa käyttöön `train.py` scriptiä hivenen muokkaamalla.

Lisätään import:

```
from taivasnet.models import NeuroseModel
```

Muutetaan oletuksena käytettävä malli. Ennen:

```
    # create the model
    model = TwoLayerModel()
```

Jälkeen:

```
    # create the model
    model = NeuroseModel()
```

Kokeillaan, toistuvatko hitausongelmat. Käytetään samaa `learning rate` parametria, joka `mnist_example.py` scriptissä on käytössä.

Aluksi vaikuttaa, ettei verkko alkaisi oppimaan, mutta n. 10 epochin jälkeen alkaa tapahtua.

```
$ time ./train.py --epoch 20 --lr=0.05
- Training model for 20 epoch, with learning rate 0.05
Epoch Train loss   Valid loss   Train acc Valid acc
0     2.3051868312 2.3023939188 0.0875000 0.1064000
1     2.3055937898 2.3019687203 0.0875000 0.1064000
2     2.3058198081 2.3019310655 0.0875000 0.1064000
3     2.3059008579 2.3019262902 0.0875000 0.1064000
4     2.3059206759 2.3019173400 0.0875000 0.1064000
5     2.3059160116 2.3019020755 0.0875000 0.1064000
6     2.3058993915 2.3018802829 0.0875000 0.1064000
7     2.3058719264 2.3018486405 0.0875000 0.1064000
8     2.3058279127 2.3017993303 0.0875000 0.1064000
9     2.3057499374 2.3017134369 0.0875000 0.1064000
10    2.3055901001 2.3015372191 0.0875000 0.1064000
11    2.3051280855 2.3010330513 0.0875000 0.1064000
12    2.3025266145 2.2982405822 0.0875000 0.1064000
13    2.0761554755 2.0746136824 0.2250000 0.2060000
14    1.6548844260 1.6047095033 0.3125000 0.3422000
15    1.4740034266 1.4013679311 0.4250000 0.4357000
16    1.2260494548 1.1851681857 0.5375000 0.5742000
17    0.8487759360 0.8429157940 0.7625000 0.7301000
18    0.4853546088 0.5140690554 0.8875000 0.8455000
19    0.3581256915 0.3970673611 0.9250000 0.8832000

real	0m36,165s
user	1m27,896s
sys	0m40,335s
```

Koko hommaan (20 epochia) meni n. 36 sekuntia.

#### Miksi alkuun pääsemisessä kestää niin kauan?

Ongelman ydin lienee neuroverkkojen painojen alustamisessa. Mitä syvempi verkko, sitä huonommin se treenaantuu.

[Koodissani](https://github.com/ikanher/numpy-MNIST/blob/4ab890c45143bc21b1fbbc7a981846f18931e362/taivasnet/taivasnet/layers.py#L55) sattui olemaan aiempin kokeilujen vuoksi niin kutsuttu [Xavier](https://theneuralperspective.com/2016/11/11/weights-initialization/) painojen alustaminen, joten tein pikaisen muutoksen ja otin sen käyttöön.

Nyt alkoi tapahtua:

```
$ time ./train.py --epochs 10 --lr=0.05
- Training model for 10 epoch, with learning rate 0.05
Epoch Train loss   Valid loss   Train acc Valid acc
0     0.2628765052 0.2670100218 0.9125000 0.9197000
1     0.1374972283 0.1929956231 0.9875000 0.9436000
2     0.0859377807 0.1630642919 0.9875000 0.9532000
3     0.0608347140 0.1466086614 1.0000000 0.9576000
4     0.0449785351 0.1361851495 1.0000000 0.9615000
5     0.0353370207 0.1295882677 1.0000000 0.9630000
6     0.0285742223 0.1244571125 1.0000000 0.9642000
7     0.0229576191 0.1201078937 1.0000000 0.9658000
8     0.0186848877 0.1166853159 1.0000000 0.9665000
9     0.0152841417 0.1136260188 1.0000000 0.9669000

real	0m19,143s
user	0m45,272s
sys	0m21,713s
```

Verkko alkaa oppimaan heti ensimmäisen epochin aikana!

Mitä syvempi verkko on, sitä suurempi merkitys on painojen alustamisella. Erityisen syviin verkkoihin suositellaankin käytettäväksi [Kaiming](https://arxiv.org/pdf/1502.01852v1.pdf) painojen alustamista.

Tämän ei kuitenkaan nopeuta alkuperäistä Neurosen koodia (vaikka se saattaisikin alkaa oppia nopeammin), vaan vika lienee jossain muualla. En tämän raportin puitteissa lähde kuitenkaan selvittämään missä ongelman ydin on (veikkaus on, että jostain löytyy for looppeja sen sijaan, että käytettäisiin vektorisoituja operaatioita).

## Mitä yhteistä?

Tarkastellaan seuraavaksi molempia kirjastoja etsien niistä löytyviä yhtäläisyyksiä ja eroja.

### Kerrokset (Layerit)

Molemmissa kirjastoissa on käytössä samoja layereitä. _Linear_, _Softmax_ ja _ReLU_. Lisäksi omassa versiossani olen toteuttanut myös _Dropout_ layerin, jota käytetään regularisoinnissa. Oma mallini alkoi [overfittaamaan](https://en.wikipedia.org/wiki/Overfitting) hyvinkin nopeasti, joten yksi tai useampi regularisointi metodi on tarpeeseen. Palaan asiaan, miksi luulen, ettei tätä ole Neurosessa tehty, puhuessani toteutuksien eroista.

### Neuroverkko abstrahointi

Molemmissa toteutuksissa on vastaavanlainen neuroverkkoa edustava olio. `Neurose` periyttää tästä oliosta erilaiset käytettävät mallit. `Taivasnet` sen sijaan antaa mallin neuroverkko-oliolle parametrinä.

Edelleen, molemmissa toteutuksissa on backpropagation ja loss funktion kutsuminen annettu neuroverkko-olion vastuulle.

### Optimoija

Molemmissa toteuksissa on käytössä _mini-batch_ Stochastic Gradient Descent, joka onkin yleisimpiä - ja helpoiten toteuttavissa olevia optimointialgoritmeja. Helpommaksi muodostuu enää _batch_ SGD, joka käyttää kaiken treenaus datan kerralla tai sitten _online_ SGD, joka käy jokaisen treenaamisen käytettävän samplen yksi kerrallaan läpi. Molemmat näistä jälkeen mainituista tavoista ovat kuitenkin huomattavasti hitaampi kuin mini-batch versio.

## Mitä eroja?

### Datan käyttö

Suurimpana erona silmääni osuu, ettei `Neurose` käytä osaa treenaus-datasta validointiin. Neuroverkoissa on erittäin tärkeää, että varsinanen testidata pidetään erillisenä, eikä sitä käytetä mallin tuunaamiseen lainkaan.

Omassa toteutuksessani jokaisen epochin jälkeen määritetään sekä treenaus häviö, että validointi häviö. Validaatio-dataksi on otettu 20% treenausdatasta, joka tuntuu olevan alan standardi.

Vertailemalla treenaus ja validaatio häviöitä saadaan erittäin tärkeää tietoa siitä, onko mallit alkanut overfittaamaan (treenaushäviö on huomattavasti pienempi kuin validaatiohäviö) tai mahdollisesti underfittaamaan (validaatiohäviö on huomattavasti pienempi kuin treenaushäviö).

Tämä lieneekin syy siihen, ettei `Neurosen` toteutuksessa ole käytetty minkäänlaista regularisaatiota - ei ole ollut tietoa, että malli overfittaa reilusti.

### Kerrokset (Layerit)

Kerrosten abstrahointi ei ole samalla tavalla toteuttu. Omassa totetuksessani kaikki layerit löytyvät `layers.py` moduulista ja loss funktiot `losses.py` moduulista.

Neurosen toteutuksessa sen sijaan `layers.py` sisältää ainoastaan `Linear` layerin.

Muut layerit, aktivointifunktiot ja häviöfunktiot löytyvät `functions.py` moduulista, jossa ne periyttävät `DifferentiableFunction` olion. Näin ollen muutet layerit, paitsi `Linear` osaavat määrittää oman derivaattansa.

Kummallista kyllä, jokaiselle layerille annetaan parametrina myös itse verkko. Tämä ehkä backpropagation toteutuksen takia? Mutta se tekee tilanteen hieman oudoksi, sillä nyt neuroverkko-olio tietää layereista ja layerit neuroverkko-oliosta.

### Neuroverkko abstrahointi

`Taivasnet` toteutuksessa jokainen layer tietää oman derivaattansa (backward -metodi). `Neurose` toteuksessa tämä vaihe on ilmeisesti sisällytetty neuroverkko-olion backpropagation vaiheeseen. `Taivasnet` sen sijaan kutsuu vain vuorollaan jokaisen layerin `backward` metodia, saaden sieltä ulos tarvittavan gradientin.

### Gradientin määrittäminen

Kuten edellisessä osuudessa todettiin, niin omassa toteutuksessani, jokainen layer tietää oman gradientinsa.

Tässä on yksi suurimpia eroja, josta en ole varma kumpi metodi olisi parempi.

Omassa toteutuksessani `CrossEntropy` loss ei tiedä mitään oman derivaattansa laskemisesta, vaan tämä on siirretty suoraan `Softmax` layerin vastuulle. Tätä tuli kyllä mietittyä devatessa, että mihin se olisi paras laittaa ja päädyin nykyiseen toteutukseen, koska sen generalisointi oli helpompi toteuttaa.

Toisaalta kuulostaa kyllä erittäin fiksultakin, että häviöfunktio tietää oman derivaattaansa, etenkin kun lisätään uusia häviöfunktioita.

### Optimoija

Omassa toteutuksessani optimoja on eriytetty omaan moduuliinsa SGD, joka saa paremetreinä optimoitavan verkon, treenausdatan lataajan sekä mini-batchin koon. Optimoinnin toteuttaa `SGD.fit` metodi, joka saa parametrinaan epochien määrän sekä oppimisnopeuden (learning rate).

Neurosessa optimoijaa eli ole abstrahoitu, vaan se on kirjotettu suoraan skriptimäisenä koodina esimerkiksi `mnist_example.py` scriptiin. Mini-batchin koko annetaan Dataloaderille parametrina. Ja ehkä hieman eriskummallisesti oppimisnopeus annetaan itse neuroverkko-oliolle. Epochien määrä koodataan luonnollisesti erikseen kirjoitetun optimointi-loopin yhteyteen.

## Mitä opin?

Tämä oli tavattoman hieno harjoitus. Oli upeaa päästä lukemaan koodia, joka tekee käytännössä samaa asiaa kuin omani.

### Moduuleihin jako

Ehkä parempi tapa jakaa koodi moduuleihin olisi toteuttaa `activations.py`, joka sisältää aktivointifunktiot, `layers.py`, joka sisältää layerit ja `losses.py`, joka sisältää häviön laskemiseen käytetyt funktiot.

### Dataloader tietoinen mini-batchin koosta

Ehdottomasti toteuttamisen arvoinen idea. Itseasiassa mietin tätä jo alkuvaiheessa, mutta sitten jäi hieman selväksi miten toteutan sen sekä treenaus-, että validointidatalle. Neurosen koodissa käytetään kahta eri dataloaderia. Tämä toteutus voisi olla järkevä, toinen treenausdatalle ja toinen validaatiodatalle.

### Missä gradientit tulisi laskea?

~~Olen edelleen sitä mieltä, että jokaisen layerin/aktivointifunktion tulisi osata laskea oma gradientinsa.~~

~~Ongelmia kuitenkin tuottaa (klassifiointiaongelmien yhteydessä, jossa on usein käytössä Softmax aktivointifunktio, sekä CrossEntropy häviöfunktio) se, että näillä on yhteinen derivaatta. Jos siirrän CrossEntropyn derivaatan laskemisen CrossEntropy moduulin vastuulle, niin kuinka Softmax funktio derivaattaa tulisi käyttää?~~

Vaikuttaa siltä, että paras paikka häviön gradientin laskemiseen on kuitenkin häviöfunktio itse. Päätinkin Neurosen toteutuksesta inspiroituneena refaktoroida oman koodini tällä tavalla. Nyt uusien häviöfunktioiden lisääminen pitäisi olla huomattavasti yksinkertaisempaa.

