import os
import subprocess
import sys
import venv

def run_cmd(command):
    print(f"\n➡️ Exécution : {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print("❌ Erreur lors de l'exécution.")
        sys.exit(1)

def check_python():
    print("🔍 Vérification de Python...")
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ est requis.")
        sys.exit(1)
    print("✔ Python OK")

def install_system_dependencies():
    print("\n📦 Installation des dépendances système...")
    run_cmd("sudo apt update")
    run_cmd("sudo apt install -y python3 python3-pip python3-venv libatlas-base-dev libhdf5-dev libhdf5-serial-dev libqtgui4 libqt4-test libilmbase-dev libopenexr-dev libgstreamer1.0-dev")

    # OpenCV optimisé Raspberry Pi
    run_cmd("sudo apt install -y python3-opencv")

def create_virtual_env():
    print("\n📂 Création de l'environnement virtuel...")
    if not os.path.exists("venv"):
        venv.create("venv", with_pip=True)
    print("✔ venv créé")

def install_python_dependencies():
    print("\n🐍 Installation des dépendances Python...")
    run_cmd("./venv/bin/pip install numpy")
    run_cmd("./venv/bin/pip install tensorflow==2.9.0")  # Version stable ARM
    run_cmd("./venv/bin/pip install opencv-python")
    run_cmd("./venv/bin/pip install pandas")

def end_message():
    print("\n✅ Installation terminée !")
    print("Pour lancer la détection :")
    print("source venv/bin/activate")
    print("python3 emotion_detection.py")

if __name__ == "__main__":
    check_python()
    install_system_dependencies()
    create_virtual_env()
    install_python_dependencies()
    end_message()
