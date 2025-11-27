"""Simple test for backend.core.analyze_image in mock mode.

This test does not require heavy ML dependencies; it calls analyze_image with
`use_mock=True` so it runs quickly and deterministically.
"""
from backend.core import analyze_image


def main():
    print('Running core.analyze_image in mock mode...')
    result = analyze_image(b'', use_mock=True)
    print('Result:')
    import json
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
