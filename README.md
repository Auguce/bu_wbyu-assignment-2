# bu_wbyu-assignment-2
# KMeans Clustering Visualization Web Application

This is a dynamic web application that visualizes the KMeans clustering algorithm using Flask and JavaScript. It allows users to generate random datasets, select different initialization methods for the centroids, and see the step-by-step clustering process.

## Features
- **Random Dataset Generation:** Generate random datasets to test different clustering algorithms.
- **Multiple Initialization Methods:** Supports Random, Farthest First, KMeans++, and Manual centroid selection.
- **Run to Convergence:** Allows the user to run the KMeans algorithm to convergence.
- **Step Through KMeans:** Provides a step-by-step execution of the KMeans clustering process.
- **Manual Initialization:** Users can manually select the initial centroids on the canvas.
- **Reset and Undo:** Reset the entire clustering process or undo the last manual centroid selection.
- **Warnings for Empty Clusters:** Alerts the user if any clusters are empty during the process.

## Demo Video
For a demonstration of the application's functionality, please watch the video linked below:

[![KMeans Clustering Visualization Demo](https://youtu.be/Y-4Mm76Eyew)](https://youtu.be/Y-4Mm76Eyew)

*(Replace `YOUR_VIDEO_ID_HERE` with the actual video ID after uploading your video to YouTube.)*

## Project Structure
- `app.py`: The main Flask application that handles routes, clustering operations, and API endpoints.
- `kmeans.py`: Contains the custom KMeans algorithm implementation, with support for different initialization methods.
- `index.html`: The front-end HTML file containing the JavaScript logic for interacting with the clustering algorithm.
- `Makefile`: Automates the setup and execution of the web application.
- `requirements.txt`: Lists all Python dependencies for the project.

## Getting Started
### Prerequisites
- Python 3.x installed on your system.
- `pip` for managing Python packages.

### Installation and Running
1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
   cd YOUR_REPOSITORY_NAME
