from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    # name : str = "tanish" # Default value
    name : str
    age : Optional[int] = None
    email : EmailStr
    cgpa : float = Field(gt = 0, lt=10, default=8, description= "Decimal value representing the cgpa of the student")


new_student = Student(**{"name" : "Tanish",'age' : 22, 'email' :'abc@gmail.com'})
new_student2 = Student(**{"name" : "Tanish", "age" : "22", 'email' : "abc@gmail.com"}) # Type Coercion ; implicit conversion of datatypes
# new_student3 = Student(**{"name" : "Tanish", "email" : "abc"}) # It will throw error
new_student3 = Student(**{"name" : "Tanish",'email' : "tanish@gmail.com"})
# new_student4 = Student(**{"name" : "Tanish", 'email' : "tanish@gmail.com", 'cgpa' : 30}) # It will throw error
new_student4 = Student(**{"name" : "Tanish", 'email' : "tanish@gmail.com", 'cgpa' : 9})

print(new_student)
print(new_student2)
print(new_student3)
print(new_student4)