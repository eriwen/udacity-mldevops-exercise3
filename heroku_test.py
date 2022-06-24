import requests

response = requests.get('https://fathomless-wildwood-07611.herokuapp.com/')

print(response.status_code)
print(response.json())