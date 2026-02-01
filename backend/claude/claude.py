import anthropic
from display_data.display import match_fires_by_date

# Exvironment variable 'ANTHROPIC_API_KEY' needs to be set

client = anthropic.Anthropic()

def sendPrompt(prompt: str, max_tokens: int = 1000) -> str:

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return message.content

def get_historical_summary(year, max_tokens = 500) -> str:
    fire_data = match_fires_by_date(year)
    prompt = f"""Generate a concise historical summary of this wildfire incident:

    Fire Name: {fire_data['name']}
    Counties: {fire_data['county']}
    Start Date: {fire_data['started']}
    Total Acres Burned: {fire_data['acres_burned']:,}
    Tracked Snapshots: {fire_data.get('summary', {}).get('total_snapshots', 0)}

    Please provide a 2-3 sentence summary covering the fire's significance, impact, and any notable characteristics.
    
    Do not use markdown."""
    
    response = sendPrompt(prompt, max_tokens)
    # print(response[0].text)
    return response[0].text if response else "Unable to generate summary"

if __name__ == "__main__":
    # Test historical summary
    summary = get_historical_summary(2020)
    print(summary)