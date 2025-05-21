# --- Data Types ---
# Numbers
integer_var = 10
float_var = 3.14
complex_var = 2 + 3j
print(f"Integer: {integer_var}, Type: {type(integer_var)}")
print(f"Float: {float_var}, Type: {type(float_var)}")
print(f"Complex: {complex_var}, Type: {type(complex_var)}")

# Strings
string_var = "Hello Python!"
multiline_string = """This is a
multi-line string."""
print(f"String: {string_var}, Length: {len(string_var)}")
print(f"Formatted string: Your integer is {integer_var}") # f-string

# Booleans
bool_true = True
bool_false = False
print(f"Boolean True: {bool_true}, Type: {type(bool_true)}")

# --- Lists (Mutable, Ordered) ---
my_list = [1, "apple", 3.14, True, 1] # Can have mixed types
print(f"Original list: {my_list}")
my_list.append("banana") # Add item to end
print(f"After append: {my_list}")
print(f"First element: {my_list[0]}")
print(f"Slice (elements 1 to 3): {my_list[1:4]}")
my_list[1] = "orange" # Modify element
print(f"After modification: {my_list}")

# --- Tuples (Immutable, Ordered) ---
my_tuple = (10, "grape", 2.5)
print(f"Tuple: {my_tuple}")
print(f"First element of tuple: {my_tuple[0]}")
# my_tuple[0] = 20 # This would cause a TypeError because tuples are immutable

# --- Dictionaries (Mutable, Unordered prior to Python 3.7, Ordered in 3.7+) ---
# Keys must be unique and immutable (e.g., strings, numbers, tuples)
my_dict = {"name": "Alice", "age": 30, "city": "New York"}
print(f"Dictionary: {my_dict}")
print(f"Name: {my_dict['name']}")
my_dict["email"] = "alice@example.com" # Add new key-value pair
print(f"After adding email: {my_dict}")
my_dict["age"] = 31 # Update existing value
print(f"After updating age: {my_dict}")
print(f"Keys: {my_dict.keys()}")
print(f"Values: {my_dict.values()}")
print(f"Items: {my_dict.items()}")

# --- Sets (Mutable, Unordered, No Duplicates) ---
my_set = {1, 2, 3, "hello", 2, "world", 1} # Duplicates are automatically removed
print(f"Set: {my_set}")
my_set.add(4)
print(f"After adding 4: {my_set}")
set1 = {1, 2, 3}
set2 = {3, 4, 5}
print(f"Union: {set1.union(set2)}")
print(f"Intersection: {set1.intersection(set2)}")

# --- Conditionals (if, elif, else) ---
temperature = 25
if temperature > 30:
    print("It's hot!")
elif temperature > 20:
    print("It's warm.")
else:
    print("It's cool or cold.")

# --- Loops ---
# For loop (iterating over a sequence)
print("Looping through a list:")
for item in my_list:
    print(item)

print("Looping with range:")
for i in range(5): # 0 to 4
    print(i)

print("Looping through dictionary items:")
for key, value in my_dict.items():
    print(f"{key}: {value}")

# While loop
count = 0
print("While loop:")
while count < 3:
    print(f"Count is {count}")
    count += 1

# --- Functions ---
def greet(name):
    """This function greets the person passed in as a parameter."""
    return f"Hello, {name}!"

message = greet("ML Engineer")
print(message)

def calculate_area(length, width):
    """Calculates the area of a rectangle."""
    if length < 0 or width < 0:
        return "Dimensions cannot be negative."
    return length * width

area1 = calculate_area(5, 4)
area2 = calculate_area(-2, 4)
print(f"Area 1: {area1}")
print(f"Area 2: {area2}")

# Function with default arguments
def power(base, exponent=2):
    """Raises base to the power of exponent."""
    return base ** exponent

print(f"3 squared: {power(3)}")
print(f"2 cubed: {power(2, 3)}")

# Function with variable number of arguments
def print_all(*args): # *args collects extra positional arguments into a tuple
    print("Arguments passed to *args:")
    for arg in args:
        print(arg)

print_all(1, "test", True)

def print_info(**kwargs): # **kwargs collects extra keyword arguments into a dictionary
    print("Keyword arguments passed to **kwargs:")
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Bob", age=25, country="Canada")

# --- Lambda Functions (Anonymous Functions) ---
# Useful for short, simple operations
add = lambda x, y: x + y
print(f"Lambda addition: 5 + 3 = {add(5, 3)}")

numbers = [1, 2, 3, 4, 5]
# Using lambda with map() to square each number
squared_numbers = list(map(lambda x: x**2, numbers))
print(f"Squared numbers (using map and lambda): {squared_numbers}")

# Using lambda with filter() to get even numbers
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
print(f"Even numbers (using filter and lambda): {even_numbers}")

# --- List Comprehensions ---
# Concise way to create lists
# Basic syntax: [expression for item in iterable]
squares = [x**2 for x in numbers]
print(f"Squares (list comprehension): {squares}")

# With a condition: [expression for item in iterable if condition]
evens_comp = [x for x in numbers if x % 2 == 0]
print(f"Evens (list comprehension): {evens_comp}")

# Example: Create a list of (number, square) tuples
num_square_pairs = [(x, x**2) for x in range(1, 6)]
print(f"Number-square pairs: {num_square_pairs}")

# --- Dictionary Comprehensions ---
# Concise way to create dictionaries
# Basic syntax: {key_expression: value_expression for item in iterable}
names = ["Alice", "Bob", "Charlie"]
name_lengths = {name: len(name) for name in names}
print(f"Name lengths (dict comprehension): {name_lengths}")

# With a condition
squared_dict = {x: x**2 for x in numbers if x > 2}
print(f"Squared dict for numbers > 2: {squared_dict}")

# --- Set Comprehensions ---
# Concise way to create sets
unique_squares = {x**2 for x in [-1, 1, -2, 2, 3]}
print(f"Unique squares (set comprehension): {unique_squares}")

# --- Classes and Objects ---

class Dog:
    # Class attribute (shared by all instances)
    species = "Canis familiaris"

    # Initializer / Instance attributes
    def __init__(self, name, age, breed="Unknown"):
        self.name = name  # Instance attribute
        self.age = age    # Instance attribute
        self.breed = breed  # Instance attribute

    # Instance method
    def description(self):
        return f"{self.name} is {self.age} years old and is a {self.breed}."

    # Another instance method
    def speak(self, sound):
        return f"{self.name} says {sound}!"

    # Dunder method (string representation of the object)
    def __str__(self):
        return f"Dog(name='{self.name}', age={self.age}, breed='{self.breed}')"

# Creating (instantiating) objects of the Dog class
dog1 = Dog("Buddy", 3, "Golden Retriever")
dog2 = Dog("Lucy", 5) # Breed will be "Unknown"

print(dog1.name)         # Accessing instance attribute
print(dog2.breed)        # Accessing instance attribute
print(Dog.species)       # Accessing class attribute

print(dog1.description())
print(dog2.speak("Woof"))
print(str(dog1)) # Uses the __str__ method
print(dog1)      # print() automatically calls __str__ if defined

# --- Example: Simple Dataset Class (Conceptual for ML) ---
class SimpleDataset:
    def __init__(self, features, labels):
        if len(features) != len(labels):
            raise ValueError("Features and labels must have the same number of samples.")
        self.features = features # Typically a list of lists, or a NumPy array later
        self.labels = labels     # Typically a list or a NumPy array

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, index):
        """Allows indexing to get a specific sample (features, label)."""
        return self.features[index], self.labels[index]

    def info(self):
        print(f"Dataset contains {len(self)} samples.")
        if len(self) > 0:
            num_features = len(self.features[0]) if isinstance(self.features[0], list) else 1
            print(f"Number of features per sample: {num_features}")

# Example usage of SimpleDataset
# Imagine these are rows of data for an ML model
sample_features = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
sample_labels = [0, 1, 0, 1] # e.g., 0 for class A, 1 for class B

my_dataset = SimpleDataset(sample_features, sample_labels)

print(f"Number of samples in dataset: {len(my_dataset)}")
first_sample = my_dataset[0]
print(f"First sample features: {first_sample[0]}, label: {first_sample[1]}")
my_dataset.info()

# Example of accessing another sample
feature_vector, label = my_dataset[2]
print(f"Sample 2 - Features: {feature_vector}, Label: {label}")

# This is a very basic dataset holder. In ML, you'd often use Pandas DataFrames
# or specific library structures (like PyTorch's Dataset or TensorFlow's tf.data).



import pandas as pd

# --- Pandas Series (1D labeled array) ---
# Creating a Series from a list
s_data = [10, 20, 30, 40, 50]
s_index = ['a', 'b', 'c', 'd', 'e']
my_series = pd.Series(data=s_data, index=s_index)
print("Pandas Series:")
print(my_series)
print(f"Value at index 'c': {my_series['c']}")
print(f"Values greater than 25:\n{my_series[my_series > 25]}")

# Creating a Series from a dictionary
dict_data = {'x': 100, 'y': 200, 'z': 300}
series_from_dict = pd.Series(dict_data)
print("\nSeries from dictionary:")
print(series_from_dict)

# --- Pandas DataFrames (2D labeled data structure with columns of potentially different types) ---
# Creating a DataFrame from a dictionary of lists
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 30, 35, 28, 22],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
    'Salary': [70000, 80000, 90000, 75000, 65000]
}
df = pd.DataFrame(data)
print("\nPandas DataFrame:")
print(df)

# --- Importing Data ---
# Make sure you have a CSV file named 'sample_data.csv' in the same directory
# or provide the full path.
# Example 'sample_data.csv' content:
# id,feature1,feature2,target
# 1,0.5,1.2,0
# 2,0.3,0.9,1
# 3,0.7,1.5,0
# 4,0.1,0.3,1
# 5,NA,1.1,0

# Create a dummy CSV for demonstration if it doesn't exist
try:
    with open('sample_data.csv', 'w') as f:
        f.write("id,feature1,feature2,target\n")
        f.write("1,0.5,1.2,0\n")
        f.write("2,0.3,0.9,1\n")
        f.write("3,0.7,1.5,0\n")
        f.write("4,0.1,0.3,1\n")
        f.write("5,,1.1,0\n") # Note the missing value (NA)
        f.write("6,0.6,1.3,1\n")
    print("\n'sample_data.csv' created for demonstration.")
except IOError:
    print("\nCould not create 'sample_data.csv'. Please ensure you have write permissions or create it manually.")


# Load data from a CSV file
try:
    df_from_csv = pd.read_csv('sample_data.csv')
    print("\nDataFrame loaded from CSV:")
    print(df_from_csv)
except FileNotFoundError:
    print("\nError: 'sample_data.csv' not found. Please create it or place it in the correct directory.")


# --- Exploring Datasets ---
if 'df_from_csv' in locals(): # Check if df_from_csv was loaded successfully
    print("\n--- Exploring the CSV DataFrame ---")
    # Display the first N rows (default is 5)
    print("\nHead (first 5 rows):")
    print(df_from_csv.head())

    # Display the last N rows
    print("\nTail (last 3 rows):")
    print(df_from_csv.tail(3))

    # Get the dimensions of the DataFrame (rows, columns)
    print(f"\nShape of the DataFrame: {df_from_csv.shape}")

    # Get column names
    print(f"\nColumn names: {df_from_csv.columns.tolist()}")

    # Get data types of each column
    print("\nData types of columns (dtypes):")
    print(df_from_csv.dtypes)

    # Get a concise summary of the DataFrame
    print("\nInfo (summary of DataFrame):")
    df_from_csv.info()

    # Get descriptive statistics for numerical columns
    print("\nDescriptive statistics (describe()):")
    print(df_from_csv.describe())

    # --- Selecting Data ---
    # Select a single column (returns a Series)
    print("\nSelecting 'feature1' column:")
    print(df_from_csv['feature1'])

    # Select multiple columns (returns a DataFrame)
    print("\nSelecting 'id' and 'target' columns:")
    print(df_from_csv[['id', 'target']])

    # Select rows by label (index) using .loc
    # (assuming 'id' is not the index yet, so it uses default integer index)
    print("\nSelecting row with index 2 (using .loc):")
    print(df_from_csv.loc[2]) # Selects the third row (0-indexed)

    print("\nSelecting rows 0 to 2 and columns 'feature1', 'feature2' (using .loc):")
    print(df_from_csv.loc[0:2, ['feature1', 'feature2']]) # Inclusive slice for .loc

    # Select rows by integer position using .iloc
    print("\nSelecting row at integer position 2 (using .iloc):")
    print(df_from_csv.iloc[2]) # Selects the third row

    print("\nSelecting rows 0 to 2 (exclusive for end) and columns 1 to 2 (exclusive for end) (using .iloc):")
    print(df_from_csv.iloc[0:3, 1:3]) # Exclusive slice for .iloc

    # Conditional selection
    print("\nRows where 'target' is 1:")
    print(df_from_csv[df_from_csv['target'] == 1])

    print("\nRows where 'feature2' > 1.0 and 'target' is 0:")
    print(df_from_csv[(df_from_csv['feature2'] > 1.0) & (df_from_csv['target'] == 0)])
else:
    print("\nSkipping CSV DataFrame exploration as 'sample_data.csv' was not loaded.")


