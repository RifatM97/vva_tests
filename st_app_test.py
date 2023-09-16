import streamlit as st

url1 = "https://d-id-talks-prod.s3.us-west-2.amazonaws.com/auth0%7C64aeb88aee6719d9da99fa67/tlk_-b2dNTz8aeuk_D_eGcNMY/1689241549987.mp4?AWSAccessKeyId=AKIA5CUMPJBIK65W6FGA&Expires=1689327961&Signature=xMlW9mLI2v5%2FgCaEPHMyLGYYJPI%3D&X-Amzn-Trace-Id=Root%3D1-64afc7d9-2c2ad170774090745639c40e%3BParent%3Da2d0585439b81aff%3BSampled%3D1%3BLineage%3D6b931dd4%3A0"
url2 = "https://d-id-talks-prod.s3.us-west-2.amazonaws.com/auth0%7C64aeb88aee6719d9da99fa67/tlk_Ed45cYIudjIqjMMT45S6K/1689239704826.mp4?AWSAccessKeyId=AKIA5CUMPJBIK65W6FGA&Expires=1689326112&Signature=bNGXB0PiOfsO0H6ZWrtgaDLXXlQ%3D&X-Amzn-Trace-Id=Root%3D1-64afc09f-46021941157bfd8d22cb0ca5%3BParent%3Dfc8fd7ced94e9ff1%3BSampled%3D1%3BLineage%3D6b931dd4%3A0"
st.title('🤖 VVA Bot Test')
st.video(url2)