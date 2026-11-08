import json
with open ('func.json', 'r') as file:
    json_object = json.load(file)

# for item in json_object:
#     print(item)
#     print('')

# with open ('test.json', 'w') as file:
#     json.dump({'"abc": 1'}, file)

# The problem is that file objects don't have a 'keys' attribute
# We need to load the JSON first and then access its keys
# with open('func.json', 'r') as file:
#     data = json.load(file)
#     print(data[0].keys())


with open ('func.json', 'r') as file:
    data = json.load(file)
    print(data[0].keys())
