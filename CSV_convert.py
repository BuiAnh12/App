# Open the input text file for reading
with open('graph.txt', 'r') as txt_file:
    lines = txt_file.readlines()

# Initialize variables to store data
data = []
current_c = None
accuracy = None
f1_score = None

# Iterate through the lines and extract data
for line in lines:
    line = line.strip()
    if line.startswith("C = "):
        # When a new 'C' value is encountered, store the previous data (if any)
        if current_c is not None:
            data.append([current_c, accuracy, f1_score])
        current_c = int(line.split("C = ")[1])
    elif line.startswith("Normal Accuracy: "):
        accuracy = float(line.split("Normal Accuracy: ")[1].rstrip("%"))
    elif line.startswith("F1 Score: "):
        f1_score = float(line.split("F1 Score: ")[1].rstrip("%"))

# Append the last set of data (if any)
if current_c is not None:
    data.append([current_c, accuracy, f1_score])

# Open a CSV file for writing
with open('output.csv', 'w') as csv_file:
    # Write a header row
    csv_file.write("C,Normal Accuracy,F1 Score\n")
    
    # Write the data rows
    for row in data:
        csv_file.write(f"{row[0]},{row[1]},{row[2]}\n")
