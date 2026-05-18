import re

def clean_text(text):

```
# Remove URLs
text = re.sub(r"http\S+", "", text)

# Remove mentions
text = re.sub(r"@\w+", "", text)

# Remove hashtags symbol only
text = re.sub(r"#", "", text)

# Remove extra whitespace
text = text.strip()

return text
```
