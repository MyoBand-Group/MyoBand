# MyoBand

A wrist-worn **surface electromyography (sEMG)** interface for recognizing hand and finger movements.

MyoBand measures electrical activity produced by the muscles in the forearm using sEMG sensors. The analogue signal is amplified using a differential amplifier and then digitized for further processing. The resulting signal is analyzed by a machine learning model, which classifies different muscle activity patterns as specific hand gestures.

The recognized gestures can then be mapped to digital actions, for example:

* 🖱️ Mouse movement and control
* ⌨️ Air typing
* 📱 Touchless phone interaction
* 🎮 Custom controls for applications or games
* 🔧 Other gesture-based interfaces

## System Overview

```text
Forearm muscles
      ↓
sEMG sensors
      ↓
Differential amplifier
      ↓
Signal processing
      ↓
Machine learning
      ↓
Gesture classification
      ↓
Digital input / application
```

## Repository Structure

The repository contains the different components required to develop and test the MyoBand system:

* **LTSpice** — Hardware, Simulations, PCB schematics, and related documentation
* **Utility** — Processing, visualization, application, and utility scripts
* **MachineLearning** — Datasets, training code, models, and gesture classification

###### TODO: Fix this README.md before launch.