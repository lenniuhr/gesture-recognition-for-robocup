# Bachelorarbeit Gestenerkennung für den RoboCup mit OpenPose und neuronalen Netzen

In dieser Arbeit wird das Framework *OpenPose* mit neuronalen Netzen kombiniert, um verschiedene Gesten anhand von Bildern zu klassifizieren.

## OpenPose

Zur Ausführung des Skripts **import-body.py** muss das Framework [*OpenPose*](https://github.com/CMU-Perceptual-Computing-Lab/openpose) installiert sein. Der Ordner *"openpose"* muss sich in diesem Verzeichnis befinden. [Zur Installation von *OpenPose*](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation.md).

## Die Datasets

Es wurde mit drei verschiedenen Datasets gearbeitet, welche sich in den Unterordnern *"images/pose-1"*, *"images/pose-2"* und *"images/pose-3"* befinden müssen. In diesen Ordnern liegen die einzelnen Bilder der Datasets im jpeg-Format vor. Die Namenskonvetion für die Bilder ist *"label - nr.jpeg"*, wobei die Nummern von 1 an aufsteigend sein müssen. Ein gültiger Name für ein Bild aus dem zweiten Dataset ist beispielsweise *"clap - 42.jpeg"*.

## Die Skripts

**import-body.py**: Verarbeitet die Bilder der Datasets mit OpenPose und importiert diese in verschiedene json-Datenbanken. Zuerst werden die OpenPose keypoints in *"db/{dataset}-raw.json"* gespeichert. Daraufhin werden aus diesen keypoints Winkel und Positionen errechnet und in *"db/{dataset}.json"* gespeichert. Die Models werden mit den Daten in *"db/{dataset}.json"* trainiert und evaluiert.

```bash
dataset = "pose-1"
file_prefixes = ["arm-up-left", "arm-up-right", "idle", "dab", "clap", "show-left", "show-right", "t-pose"]

# imports OpenPose keypoints to "pose-1-raw.json"
import_keypoints(dataset, file_prefixes)
check_raw_entries(dataset, file_prefixes)

# imports selected angles and positions to "pose-1.json"
import_body(dataset, file_prefixes, file_prefixes)
check_body_entries(dataset, file_prefixes)
```

**model-1.py**: Enthält ein Model für das erste Dataset (*"pose-1"*) mit 8 verschiedenen Gesten in 280 Bildern. Beim Ausführen des Skripts wird eine grid search durchgeführt.

**model-2.py**: Enthält ein Model für das zweite Dataset (*"pose-2"*) mit 20 verschiedenen Gesten in 1813 Bildern. Beim Ausführen des Skripts wird eine grid search durchgeführt.

**model-3.py**: Enthält ein Model für das dritte Dataset (*"pose-3"*) mit 6 verschiedenen Gesten in 540 Sequenzen aus jeweils 10 Bildern. Beim Ausführen des Skripts wird eine grid search durchgeführt.