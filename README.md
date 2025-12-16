# User Manual for Chatbot Application
## Overview
This guide will walk you through setting up and running the chatbot application. The app is built using Python 3.11 and runs inside a virtual environment called WVenv (Waste Vision Environment) to manage all the required libraries and dependencies. Follow the steps below to get started. Instruction on how to use the chatbot are described at the bottom of this document.

## Step 1: Install Python 3.11
Before running the application, make sure Python 3.11 is installed on your system.

### For Linux:

1. Open a terminal window.
2. Run the following command to install Python 3.11:

```bash 
sudo apt install python3.11
```

3. If the system cannot find the package, you may need to add a repository first. Run the following commands:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11
```

### For Windows:

1. Download and install Python 3.11 from the official Python website: https://www.python.org/downloads/
2. Make sure to check the option "Add Python to PATH" during installation.

## Step 2: Create the Virtual Environment

The application runs inside a virtual environment to keep its dependencies organized. Here’s how to create it:

### On Linux:

1. Open a terminal.
2. Navigate to the directory where the project files are located.
3. Create the virtual environment by running:
```bash
python3.11 -m venv WVenv
```

If you see an error like:
```bash
Error: Command '['/home/aiadmin/WasteVision/WVenv/bin/python3.11', '-m', 'ensurepip', '--upgrade', '--default-pip']' returned non-zero exit status 1.
```

This means some required tools are missing. To fix it, run:
```bash
sudo apt install python3.11-venv
```

Retry the virtual environment creation command from step 3.

### On Windows:
1. Open a command prompt (CMD) in the project directory.
2. Run the following command to create the virtual environment:
```bash
python -m venv WVenv
```

## Step 3: Activate the Virtual Environment
### On Linux:
1. After creating the virtual environment, activate it by running:
```bash
source WVenv/bin/activate
```

2. You should see the environment name (WVenv) in your terminal prompt, indicating that it is active.

### On Windows:
1. To activate the virtual environment, run:
```bash
WVenv\Scripts\activate
```

2. You should now see (WVenv) in your command prompt, showing the environment is active.

## Step 4: Install the Required Packages
Now that the virtual environment is active, you need to install the necessary packages that the chatbot app uses. These packages are listed in the requirements.txt file. This step is the same for both operating systems.

1. Ensure that you're inside the WVenv virtual environment.
2. Install the required packages by running:
```bash
pip install -r requirements.txt
```

This will download and install all the necessary libraries to run the chatbot.

## Step 5: Place the Data Folder in the Repository
Before running the application, make sure that the **data** containing the dataset files is placed in the root directory of the repository.

1. Locate the **data** folder in the repository
2. Copy the entire dataset you want to use into this folder. Nested folders are allowed.
3. Ensure the folder now contains all the nessasary files for the chatbot to analyse.

This step will allow the chatbot to access the data when it needs to perform data analysis and generate responses based on the available information.

## Step 5: Run the Chatbot Application
With everything set up, you can now run the chatbot application. Simply execute the following command while the virtual environment is active:

```bash
streamlit run ./main.py
```

This will start the chatbot on addres localhost:8501. Open this in your browser to access the chatbot interface.

## Step 6: Deactivate the Virtual Environment
Once you’re done, you can deactivate the virtual environment by running:
```bash
deactivate
```

## Troubleshooting
If you encounter issues with installing Python or setting up the virtual environment, ensure that you have the correct permissions and try running commands with 'sudo' (on Linux) or 'Run as Administrator' (on Windows).

If the virtual environment doesn't activate or gives errors, try deleting the WVenv folder and following the steps again.


# How to Formulate User Input

To get the best results from the chatbot, follow these guidelines when formulating your questions:

---

## 1. Using Filenames

If you want the chatbot to answer your question about a specific file in the database, mention the file name in your user input. The chatbot will automatically select the correct file for analysis. For example:
```
How many orders are there in orders_export?
```
---

## 2. Add Context to Your Question

Provide as much context as possible when asking your questions. For example, instead of asking something vague like "isles in Hilversum," ask:  
```
How many total container isles are present in the city of Hilversum?
```
Adding more details helps the chatbot understand exactly what you're looking for. If you already have an idea of how SQL code could generate your answer, or if you are searching for a specific value, feel free to include that information as well.

---

## 3. Be Specific

The more specific your question is, the better the chatbot can respond. For example, if you're asking about how many of something exist in a particular area, clearly state that you're asking for unique values in that area. This makes it easier for the chatbot to reason about the exact information you need.

For example, instead of asking "How many waste containers are in the system?", ask:  
```
How many unique waste containers are present in the whole system?
```
Being specific will lead to more accurate and relevant answers.

