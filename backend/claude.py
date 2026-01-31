import anthropic

# Exvironment variable 'ANTHROPIC_API_KEY' needs to be set

client = anthropic.Anthropic()

def sendPrompt(prompt: str, max_tokens: int = 1000) -> str:

    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return message.content

if __name__ == "__main__":
    print("caluse")