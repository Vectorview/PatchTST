import csv
import math

def split_csv(input_file):
    chunk_size = 192
    test_chunk_count = 10

    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

    total_rows = len(rows)
    total_chunks = math.ceil(total_rows / chunk_size)

    test_chunks = rows[:chunk_size * 5] + rows[-chunk_size * 5:]
    train_chunks = rows[chunk_size * 5:-chunk_size * 5]

    for i in range(test_chunk_count):
        start = i * chunk_size
        end = start + chunk_size
        chunk = test_chunks[start:end]

        with open(f"test_data_{i + 1}.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(chunk)

    with open("train_data.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(train_chunks)

    print(f"Total rows: {total_rows}")
    print(f"Total chunks: {total_chunks}")
    print(f"Test data files created: {test_chunk_count}")
    print("Training data file created: train_data.csv")

input_file = 'ETTh1.csv'
split_csv(input_file)
