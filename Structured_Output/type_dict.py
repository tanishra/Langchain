from typing import TypedDict, Optional

class Person(TypedDict):
    name : str
    age : Optional[int]

new_person : Person = {"name" : "tanish", "age" : 22}
new_person2 : Person = {"name" : "tanish"}

print(new_person)
print(new_person2)