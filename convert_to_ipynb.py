import nbformat

def py_to_ipynb(py_file, ipynb_file):
    with open(py_file, 'r', encoding='utf-8') as f:
        code = f.read()

    # Create a new Jupyter Notebook
    notebook = nbformat.v4.new_notebook()

    # Add the Python code as a cell
    notebook.cells.append(nbformat.v4.new_code_cell(code))

    # Write the notebook to a file
    with open(ipynb_file, 'w', encoding='utf-8') as f:
        nbformat.write(notebook, f)

# List of files to convert
files_to_convert = [
    (r"c:\Users\novee\House Prediction\AI-377\dashboard\dashboard.py", 
     r"c:\Users\novee\House Prediction\AI-377\dashboard\dashboard.ipynb"),
    (r"c:\Users\novee\House Prediction\AI-377\ml_service\model\model_training.py", 
     r"c:\Users\novee\House Prediction\AI-377\ml_service\model\model_training.ipynb"),
    (r"c:\Users\novee\House Prediction\AI-377\ml_service\api\app.py", 
     r"c:\Users\novee\House Prediction\AI-377\ml_service\api\app.ipynb"),
    (r"c:\Users\novee\House Prediction\AI-377\ml_service\data_processing.py", 
     r"c:\Users\novee\House Prediction\AI-377\ml_service\data_processing.ipynb")
]

# Convert each file
for py_file, ipynb_file in files_to_convert:
    py_to_ipynb(py_file, ipynb_file)
    print(f"Converted {py_file} to {ipynb_file}")