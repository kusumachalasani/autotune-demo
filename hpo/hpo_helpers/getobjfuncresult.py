import json, csv
import os

## Calculates objective function result value 
## Input: searchspacejson , outputcsvfile from benchmark , objective function variables as string.
## Output: objective function result value.
## If no objective variables are defined, it will check if they are defined in searchspace.
## Gets the objective function from searchspace and replace it with the values
## from outputcsv file to evaluate objective function result.

def calcobj(searchspacejson, inputfile, objfuncvariables):
    if not os.path.exists(inputfile):
        print("-1")
        return

    if os.path.getsize(inputfile) == 0:
        print("-1")
        return

    ## Convert the string of objective function variables defined into list
    if objfuncvariables != "":
        objfunc_variables = list(objfuncvariables.split(","))

    funcvariables = []
    with open(searchspacejson) as f:
        sdata1 = json.load(f)

        for sdata in sdata1:
            ## Get objective function
            if sdata == "objective_function":
                objf = sdata1["objective_function"]
            ## Get function variables from searchspace if defined and if objfuncvariables is empty.
            if sdata == "function_variables":
                if objfuncvariables == "":
                    funcvar = sdata1["function_variables"]
                    for fvar in funcvar:
                        for fkeys in fvar.keys():
                            if(fkeys == "name"):
                                funcvariables.append(fvar.get(fkeys))

    if objfuncvariables == "":
        objfunc_variables = funcvariables

    _, file_extension = os.path.splitext(inputfile)
    if file_extension.lower() == ".csv":
        with open(inputfile, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            csvheader = reader.fieldnames
        for row in reader:
            for x in objfunc_variables:
                for k,v in row.items():
                    if (k == x):
                        objf = objf.replace(x , v)

    if file_extension.lower() == ".json":
        with open(inputfile) as jsonfile:
            input_data = json.load(jsonfile)
        for entry in input_data:
            values = entry.get("values", {})
            for x in objfunc_variables:
                if x in values:
                    objf = objf.replace(x, str(values[x]))

    try:
        print(eval(objf))
        return
    except:
        print("-1")
        return


