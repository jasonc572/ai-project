import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

DATA_FILENAME = 'locations.json'
OPENAI_API_URL = 'https://api.openai.com/v1/chat/completions'
OPENAI_MODEL = 'gpt-3.5-turbo'
GEOCODE_API_URL = 'https://nominatim.openstreetmap.org/search'
USER_AGENT = 'ai-project/1.0'


def load_env_files():
    root = Path(__file__).resolve().parent
    load_dotenv(root / '.env')
    load_dotenv(root / 'api.env' / 'api.env')


def basic_checklist(age):
    age = int(age)
    checks = []
    if age < 1:
        checks.append('Newborn screening, immunizations, growth monitoring')
    elif age < 18:
        checks.append('Routine pediatric checkups, immunizations, vision and hearing tests')
    else:
        checks.append('Blood pressure, BMI, basic metabolic panel, lipid profile')
        if age >= 18 and age < 35:
            checks.append('Sexual health screening as appropriate, mental health check')
        if age >= 35:
            checks.append('Diabetes screening (A1C), thyroid function if symptomatic')
        if age >= 45:
            checks.append('Colorectal cancer screening discussion, cardiac risk assessment')
        if age >= 50:
            checks.append('Colonoscopy screening per guidelines, bone density discussion')
        if age >= 65:
            checks.append('Vaccinations (flu, shingles, pneumococcal), fall risk and cognitive screening')
    return checks


def prompt_age():
    while True:
        age_input = input('Enter age: ').strip()
        if not age_input:
            print('Please enter your age.')
            continue
        try:
            age = int(age_input)
            if age < 0 or age > 130:
                raise ValueError
            return age
        except ValueError:
            print('Enter a valid age between 0 and 130.')


def get_api_key():
    load_env_files()
    api_key = os.getenv('MY_API_KEY', '').strip()
    if not api_key:
        raise RuntimeError('Missing MY_API_KEY in .env, api.env/api.env, or environment')
    return api_key


def load_locations():
    root = Path(__file__).resolve().parent
    path = root / DATA_FILENAME
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def prompt_location_request():
    location = input('Enter your location (city/address or lat,lon): ').strip()
    if not location:
        raise ValueError('Location is required.')
    treatment = input('Preferred treatment type (optional): ').strip()
    return location, treatment


def is_lat_lon(location):
    try:
        lat_str, lon_str = location.split(',')
        lat = float(lat_str.strip())
        lon = float(lon_str.strip())
        return -90 <= lat <= 90 and -180 <= lon <= 180
    except Exception:
        return False


def geocode_with_opencage(location, api_key):
    url = (
        'https://api.opencagedata.com/geocode/v1/json?'
        f'q={urllib.parse.quote(location)}&key={urllib.parse.quote(api_key)}&limit=1'
    )
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.loads(response.read().decode('utf-8'))
        results = data.get('results') or []
        if not results:
            raise ValueError('No geocoding results from OpenCage.')
        first = results[0]
        geometry = first.get('geometry', {})
        return {
            'lat': float(geometry.get('lat')),
            'lon': float(geometry.get('lng')),
            'display_name': first.get('formatted', location),
        }


def geocode_with_nominatim(location):
    url = (
        f'{GEOCODE_API_URL}?q={urllib.parse.quote(location)}&format=json&limit=1'
    )
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.loads(response.read().decode('utf-8'))
        if not data:
            raise ValueError('No geocoding results from Nominatim.')
        first = data[0]
        return {
            'lat': float(first['lat']),
            'lon': float(first['lon']),
            'display_name': first.get('display_name', location),
        }


def geocode_location(location, geocode_api_key=None):
    if is_lat_lon(location):
        lat, lon = [float(part.strip()) for part in location.split(',')]
        return {'lat': lat, 'lon': lon, 'display_name': location}

    if geocode_api_key:
        try:
            return geocode_with_opencage(location, geocode_api_key)
        except Exception:
            pass

    return geocode_with_nominatim(location)


def build_recommendation_prompt(locations, user_location, user_coords, treatment):
    treatment_text = treatment or 'any available treatment'
    prompt = (
        'You are an assistant that recommends the best treatment facility from a dataset. '
        'Only use the facilities listed below. Do not invent new facilities or locations. '
        'If the user requests a treatment type, prefer facilities that offer it. '
        'If the facility is not available, it cannot be recommended. '
        'Choose the closest appropriate location based on the user location and dataset context. '
        'Return valid JSON only, with keys: location_name, address, available_treatments, selected, reason.\n\n'
        f'User input location: {user_location}\n'
        f'Parsed coordinates: {user_coords["lat"]},{user_coords["lon"]}\n'
        f'Geocoded address: {user_coords["display_name"]}\n'
        f'Requested treatment: {treatment_text}\n\n'
        'Facility dataset:\n'
        f'{json.dumps(locations, indent=2)}\n'
        'If no suitable available facility exists, return selected=false and a clear reason. '
        'Do not include any surrounding markdown or extra text.'
    )
    return prompt


def call_openai(api_key, prompt):
    payload = {
        'model': OPENAI_MODEL,
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0.0,
    }
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }
    request = urllib.request.Request(
        OPENAI_API_URL,
        data=json.dumps(payload).encode('utf-8'),
        headers=headers,
        method='POST',
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode('utf-8', errors='ignore')
        raise RuntimeError(
            f'OpenAI API error: {exc.code} {exc.reason}. ' 
            f'Response body: {error_body}'
        )


def extract_json(text):
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1:
        raise ValueError('No JSON object found in AI response')
    return json.loads(text[start:end + 1])


def recommend_treatment_location(api_key, locations):
    user_location, treatment = prompt_location_request()
    geocode_key = os.getenv('GEOCODE_API_KEY', '').strip()
    if not geocode_key and not is_lat_lon(user_location):
        if sys.stdin.isatty():
            ask_key = input('No GEOCODE_API_KEY found in environment. Enter one now or press Enter to use the fallback geocode service: ').strip()
            if ask_key:
                geocode_key = ask_key
        else:
            print('No GEOCODE_API_KEY configured; using fallback geocode service.')
    user_coords = geocode_location(user_location, geocode_key or None)
    prompt = build_recommendation_prompt(locations, user_location, user_coords, treatment)
    response = call_openai(api_key, prompt)
    message = response['choices'][0]['message']['content']
    return extract_json(message)


def run_age_checklist():
    age = prompt_age()
    print('\nSuggested checks:')
    for item in basic_checklist(age):
        print(f'- {item}')


def run_location_recommender():
    api_key = get_api_key()
    locations = load_locations()
    recommendation = recommend_treatment_location(api_key, locations)
    print('\nAI recommendation:')
    if recommendation.get('selected'):
        print(f"- Name: {recommendation.get('location_name')}")
        print(f"- Address: {recommendation.get('address')}")
        print(f"- Treatments: {', '.join(recommendation.get('available_treatments', []))}")
        print(f"- Reason: {recommendation.get('reason')}")
    else:
        print('No suitable available facility found.')
        print(f"Reason: {recommendation.get('reason')}")


def main():
    print('Medical Checklist & AI Treatment Locator')
    print('----------------------------------------')
    print('1. Age-based medical checklist')
    print('2. Find nearest available treatment location with AI')
    choice = input('Choose 1 or 2: ').strip()
    if choice == '1':
        run_age_checklist()
    elif choice == '2':
        run_location_recommender()
    else:
        print('Invalid choice. Please run the script again and choose 1 or 2.')


if __name__ == '__main__':
    main()