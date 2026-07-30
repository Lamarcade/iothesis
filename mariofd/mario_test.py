import sys

def main():
    try:
        import mario
        db = mario.load_test("IOT")
        print("MARIO load_test succeeded:", type(db))
    except Exception as e:
        print("MARIO test failed:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
