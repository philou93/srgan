"""
On a besoin de ca pour prendre en considération si on exécute le code sur gcloud ou localement.
Exemple: pour les io on doit utiliser la classe de tf sur gcloud, mais ca marche pas localement.
"""

location = ""

def set(args):
    global location
    location = args.location
