from dotenv import load_dotenv
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).parent / "keys.env")

def main():
    print("Hello from vzlotn-ae-cap-1-1!")


if __name__ == "__main__":
    main()
