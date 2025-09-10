# Automated Pest Detection and Crop Health Analysis Using Deep Learning

This project provides a comprehensive computer vision system that leverages deep learning for automated agricultural disease classification. Using the PlantVillage dataset, this project explores and implements various models to accurately identify plant diseases, contributing to precision agriculture and sustainable farming practices.

## About the Project

This project was developed as a part of a research study on automated pest detection and crop analysis. It showcases a complete workflow from data preprocessing and model training to security implementation and deployment simulation. The primary goal is to provide a robust and secure system for identifying plant diseases from images, with a high degree of accuracy.

## Dataset

The project utilizes the **PlantVillage dataset**, which contains **108,610 images** across **38 different disease classes**. The dataset is well-structured and suitable for training deep learning models for agricultural applications.

The dataset is expected to be structured as follows:

```
data/color/
├── Apple___Apple_scab/
├── Tomato___Bacterial_spot/
├── Orange___Haunglongbing/
└── ... (38 total classes)
```

## Models and Features

This project implements and compares several deep learning models:

*   **Custom CNN:** A custom-built Convolutional Neural Network with ResNet blocks and an attention mechanism, achieving over 90% accuracy.
*   **Transfer Learning:** Utilizes a pre-trained ResNet50V2 model, fine-tuned for the PlantVillage dataset.
*   **YOLOv8:** A state-of-the-art object detection model is used for a comparative analysis of classification performance.

### Key Features

*   **High Accuracy:** The models are trained to achieve high accuracy in multi-class plant disease classification.
*   **Security:** The project integrates security measures, including AES-256 encryption for data protection and adversarial robustness evaluation to defend against malicious attacks.
*   **Scalability:** The project includes a simulated deployment architecture with a RESTful API, demonstrating how the model can be scaled for real-world use.
*   **Comprehensive Analysis:** The notebook provides a detailed analysis of the dataset, model performance, and economic impact of the system.

## Getting Started

To get started with this project, follow these steps:

### Requirements
The project requires the following libraries:
```
tensorflow
numpy
pandas
scikit-learn
opencv-python
matplotlib
seaborn
cryptography
flask
boto3
ipykernel
```

### Prerequisites

*   Python 3.7 or higher
*   Jupyter Notebook or JupyterLab

### Installation

1.  **Download the project:** Unzip the project folder to your desired location.
2.  **Install dependencies:** Open a terminal or command prompt, navigate to the project directory, and run the following command to install the required libraries:

    ```bash
    pip install -r requirements.txt
    ```

### Running the Notebook

1.  **Launch Jupyter:** In your terminal, navigate to the project directory and run the following command:

    ```bash
    jupyter notebook
    ```

2.  **Open the notebook:** In the Jupyter interface, click on the `pest-detection-analysis.ipynb` file to open it.

3.  **Run the cells:** The notebook is designed to be run sequentially. You can run each cell by clicking the "Run" button or by pressing `Shift + Enter`.

    **Note:** The notebook expects the PlantVillage dataset to be in a `data/color` directory within the project folder. Please make sure you have the dataset in the correct location before running the notebook.

## Project Structure

```
.
├── .gitignore
├── pest-detection-analysis.ipynb
├── README.md
├── requirements.txt
└── outputs/
    ├── dataset_overview.png
    ├── generate_dataset_screenshots.py
    ├── random_rows_detailed.png
    └── Training Data Distribution Class, Sample Trainig Image, Dataset Split Distribution, Top 10 Classes by Sample Count.png
```

*   `pest-detection-analysis.ipynb`: The main Jupyter Notebook containing all the code, analysis, and documentation.
*   `requirements.txt`: A list of all the Python libraries required to run the project.
*   `outputs/`: This directory contains images and other output files generated during the analysis.
*   `README.md`: This file.

## Contributing

This project was developed by a student at their university. Contributions, issues, and feature requests are welcome.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
