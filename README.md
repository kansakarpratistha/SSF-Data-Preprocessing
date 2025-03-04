# SSF Data Preprocessing

## Requirements

Before you begin, make sure you have the following dependencies installed:

- `pandas`
- `owncloud`
- `numpy`

You can install these dependencies easily using the `requirements.txt` file provided:

```bash
pip install -r requirements.txt
```

## Installation
You can install the python script from the repository or clone the repository to your local machine with the following command:

```bash
git clone https://github.com/kansakarpratistha/SSF-Data-Preprocessing.git
```

## Usage
To run the processing script, navigate to the project directory:
```bash
cd SSF-Data-Preprocessing
```
then run the following command:
```bash
py process_erz.py --trench <trench_name>
```
Replace <trench_name> with one of the following:
- t1 for Trench 1
- t2 for Trench 2
- t3 for Trench 3 

## Output

For each trench, the program outputs three CSV files:

1. **Merged CSV**: Merges all time series data for the trench into a single CSV, using headers obtained from processing the trench logger configuration file and eliminating any duplicates.
2. **Processed CSV**: Handles missing values (interpolates and fills in gaps where necessary), calibrates flow values, and eliminates unwanted values.
3. **Flag CSV**: Flags flow, EC (Electrical Conductivity), and temperature values based on specific criteria.


