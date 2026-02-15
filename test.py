from transformers import pipeline

clf = pipeline(
    "text-classification", model="./modelo_fine_tuned", tokenizer="./modelo_fine_tuned"
)

# print(clf("El profesor humilla a los estudiantes"))
# print(clf("Sus comentarios son ofensivos"))
print(clf("Es el mejor docente de la universidad"))
# print(clf("acoso sexualmente a un estudiante"))
