Program zliczaj�cy mrugni�cia oczu

1. Uruchomienie programu
Aby uruchomi� program dla domy�lnej kamery video w katalogu programu uruchomi� konsol� i polecenie:

python Blinks.py

Aby wybra� plik video dodatkowo
[-v] nazwa.mp4

Dost�pne s� dwa pliki w rozmiarach 640x360:
test1.mp4 - twarz znajduj�ca si� blisko kamery
test2.mp4 - twarz oddalona od kamery

Dodatkowe opcje podczas testowania programu:
wybr�r detekcji twarzy - domy�lnie detektor z dlib, dla 'h' wyb�r klasyfikatora Haara,
kt�rego plik xml jest zawarty w pliku konfiguracyjnym w zmiennej haar_classificator
[-d] h 

skalowanie obrazu, podana warto�� zmniejsza/zwi�ksza rozmiar procentowo. Przyk�adowo aby zmniejszy� dwukrotnie 50
[-s] 50

Wyb�r algorytmu, domy�lnie oparty o granic�, dla podanej warto�ci '1' oparty o spadki
[-a] 1

3.Opis okna programu
Aby wy�wietli� wykresy z przebiegu programu do danego momentu nale�ny nacisn�� przycisk 'q'.
Aby automatycznie wy�wietli� wykresy po zako�czeniu dzia�ania programu nale�y zmienic warto�� w pliku konfiguracjnym
zmiennej plot_show na 1.

Blinks - liczba zliczonych mrugnie� w programie
Ratio - aktualna warto�� proporcji oczu
Frames - liczba klatek pokazuj�ca ile klatek zaj�o ostatnie mrugni�cie