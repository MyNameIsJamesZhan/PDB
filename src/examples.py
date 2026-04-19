from config import MIN_MULTILINES, MAX_MULTILINES

# --- Single-line bug actions ---
bug_type_examples = [
    "Delete any ONE of the following lines",
    "Add ONE line BEFORE any of the following lines",
    "Modify any ONE line of the following",
]

# --- Multiline bug actions ---
# NOTE: [design thought] Multiline actions target contiguous blocks of
# MIN_MULTILINES..MAX_MULTILINES lines. Constraining for realism: no orphaned
# indentation, no deleting control-flow headers without their bodies, etc.
_RANGE = f"{MIN_MULTILINES} to {MAX_MULTILINES}"
multiline_bug_type_examples = [
    f"Modify a contiguous BLOCK of {_RANGE} lines from the following lines. "
    "For each line in the block, you can choose to either delete it, insert a new line, or modify it by changing some tokens. "
    "Change the logic within the block while preserving indentation and code structure. "
    "Do NOT delete lines that would break indentation structure (e.g., do not delete a for/if header without its body). "
    "A deleted block should leave the remaining code syntactically plausible"
    "The final diff should show ONE contiguous block of changed lines.",
]

odc_category_probs = {
    "Assignment": 0.1,
    "Checking": 0.2,
    "Algorithm": 0.4,
    "Build/Package/Merge": 0.2,
    "Timing/Serialization": 0.1,
}

odc_categories = {
    "Assignment": {
        "Definition": "Defects where a value is initialized, defined, or mapped incorrectly.",
        "Examples": {
            "Mutability Trap": """A list, dictionary, or other mutable object used as a default argument is created only once when the function is defined. Subsequent calls to the function without that argument will modify the same object, leading to unexpected shared state across calls.
Bug Example: def add_to_list(item, my_list=[]): ...""",
            "Late Binding in Closures": """Variables in loops are bound by reference, not value. A lambda or function defined in a loop will use the final value of the loop variable when it's eventually called, not the value it had when the function was defined.
Bug Example: multipliers = [lambda x: i * x for i in range(5)]  # All lambdas will use i=4.""",
            "List Multiplication Surprise": """Using `[[]] * N` to create a list of lists results in a list containing N references to the *very same* inner list. Modifying one inner list (e.g., matrix[0].append(1)) will appear to modify all of them.
Bug Example: matrix = [[]] * 4.""",
            "Variable Shadowing": """Shadowing an outer scope variable with a local one, confusing which reference is active.
Bug Example: a class attribute self.x being shadowed by a local variable x inside a method.""",
            "Name Error": """Often caused by failing to initialize/assign a variable before use.
Bug Example: Referencing x before x is defined."""
        }
    },
    "Checking": {
        "Definition": "Defects in validation logic, conditionals, or error handling.",
        "Examples": {
            "Off-by-One Error": """A loop condition or range check is exactly one unit too high or too low.
Bug Example: if x < 10: vs. if x <= 10: .""",
            "Negation Error": """A boolean check logic is reversed or incorrect.
Bug Example: if not x == y: vs. if x == y: .""",
            "Missing or Incomplete Checks": """Runtime failures, such as Index / Key Error, or Type / Value Error, can be caused by missing or incomplete checks.
Bug Example: if key in dict and dict[key] < 3: vs. if dict[key] < 3: ; also, my_set.add([1, 2]) raises TypeError: unhashable type: 'list'""",
            "Overwriting Built-in Names": """Assigning a value to a variable name that shadows a built-in function (e.g., `list`, `sum`, `dict`) will cause `TypeError`s when the code later tries to call the original built-in function using that name.
Bug Example: list = [1, 2, 3] # Later, list((4, 5)) fails.""",
            "Variable Shadowing": """Shadowing an outer scope variable with a local one, confusing which reference is active.
Bug Example: a class attribute self.x being shadowed by a local variable x inside a method.""",
            "Chained Boolean Comparison Logic": """An expression like `if x in my_list == True:` is parsed by Python as `(x in my_list) and (my_list == True)`. The second part is almost always `False`, making the entire expression fail unexpectedly.
Bug Example: if "a" in ["a", "b"] == True: ... # This is False.""",
            "Implicit Boolean Conversion of Collections": """Checking a collection in a boolean context (e.g., `if not my_list:`) evaluates to `True` for both an empty collection (`[]`) and `None`. This can hide the important logical distinction between 'no data provided' and 'an empty set of data'.
Bug Example: if not records: ... # Runs for both records=None and records=[].""",
            "Membership Logic Flaws": """Misunderstanding how in works for specific types (e.g., iterating over a dictionary checks keys, not values).
Bug Example: if "value" in my_dict: evaluates to False even if "value" exists as a value."""
        }
    },
    "Algorithm": {
        "Definition": "Defects in the underlying recipe or method. The code implements the wrong approach for the task.",
        "Examples": {
            "Wrong Math Expression": """The calculation formula is wrong (either logical or order of operations). The fix requires rewriting the mathematical expression.
Bug Example: x = a[i] + a[j] -> x = a[i] + b[j], or x = a[i] + a[j] - 1.""",
            "Modifying a List While Iterating": """Removing or adding elements to a list while iterating over it directly disrupts the iterator's internal index. This causes the loop to skip over elements immediately following a removed item.
Bug Example: for item in my_list: my_list.remove(item).""",
            "Function Algorithm Misunderstanding": """The user misunderstood the algorithm of the function (set-based removal vs. substring removal). The fix requires choosing a different function/algorithm. For example, Assigning a value to a pandas DataFrame using chained indexing (e.g., `df[...][...] = value`) often operates on a temporary copy of the data, not the original DataFrame. The assignment may fail silently or raise a `SettingWithCopyWarning`.
Bug Example: "example.com".strip(".com") # Results in "xampl".""",
            "Function Argument Misunderstanding": """The user misunderstood the argument of the function. For example, regex quantifiers like `*` and `+` are greedy by default, meaning they match the longest possible string. This can lead to incorrect results when parsing nested structures.
Bug Example: re.match('<.*>', '<h1>title</h1>') # Matches the whole string.""",
            "Infinite Loop/Recursion": """The loop or recursion will not terminate.
Bug Example: def traverse_nodes(node):\n  for child in node.children:\n    traverse_nodes(child) # Missing if node in visited: return""",
            "Other Logical Errors": """Other complex logical errors in implementation. For example, the vanishing key paradox: if your algorithm modifies the attributes of an object while it sits inside a set/dict, it breaks the internal data structure's invariant. The object becomes irretrievable.
Bug Example: my_set = {p}; p.name = "Luigi". Now p in my_set is false""",
        }
    },
    "Build/Package/Merge": {
        "Definition": "Defects related to configuration, libraries, or version control.",
        "Examples": {
            "Invalid API call": """Code calls an API method that does not exist for a certain type of data.
Bug Example: g = df.groupby("day"); g.plot(kind="bar") # AttributeError, .plot() exists for DataFrame, Series, but not DataFrameGroupBy, Similarly, x = np.array([1, 2, 3]); x.mean(axis=1) is also invalid.""",
            "Dependency Version Conflicts": """Code relies on a specific API method that was removed in a newer version of a library (e.g., Pandas or NumPy changes).
Bug Example: AttributeError: module 'pandas' has no attribute 'Panel' (removed in modern Pandas)""",
        }
    },
    "Timing/Serialization": {
        "Definition": "Defects related to shared resources, concurrency, or the sequence of events.",
        "Examples": {
            "Serialization Issue on Outputting": """Attempting to use pickle/json to save objects that cannot serialize.
Bug Example: pickle.dumps(lambda x: x) raises PicklingError. Also, json.dumps({'date': datetime.now()}) raises TypeError.""",
            "Async Blocking": """Calling a synchronous (blocking) function inside an async def coroutine, effectively pausing the entire event loop.
Bug Example: async def f(): time.sleep(5) (Blocks all other tasks; should use await asyncio.sleep(5)."""
        }
    }
}
