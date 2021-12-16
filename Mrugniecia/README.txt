Program zliczaj¹cy mrugniêcia oczu

1. Uruchomienie programu
Aby uruchomiæ program dla domyœlnej kamery video w katalogu programu uruchomiæ konsolê i polecenie:

python Blinks.py

Aby wybraæ plik video dodatkowo
[-v] nazwa.mp4

Dostêpne s¹ dwa pliki w rozmiarach 640x360:
test1.mp4 - twarz znajduj¹ca siê blisko kamery
test2.mp4 - twarz oddalona od kamery

Dodatkowe opcje podczas testowania programu:
wybrór detekcji twarzy - domyœlnie detektor z dlib, dla 'h' wybór klasyfikatora Haara,
którego plik xml jest zawarty w pliku konfiguracyjnym w zmiennej haar_classificator
[-d] h 

skalowanie obrazu, podana wartoœæ zmniejsza/zwiêksza rozmiar procentowo. Przyk³adowo aby zmniejszyæ dwukrotnie 50
[-s] 50

Wybór algorytmu, domyœlnie oparty o granicê, dla podanej wartoœci '1' oparty o spadki
[-a] 1

3.Opis okna programu
Aby wyœwietliæ wykresy z przebiegu programu do danego momentu nale¿ny nacisn¹æ przycisk 'q'.
Aby automatycznie wyœwietliæ wykresy po zakoñczeniu dzia³ania programu nale¿y zmienic wartoœæ w pliku konfiguracjnym
zmiennej plot_show na 1.

Blinks - liczba zliczonych mrugnieæ w programie
Ratio - aktualna wartoœæ proporcji oczu
Frames - liczba klatek pokazuj¹ca ile klatek zajê³o ostatnie mrugniêcie