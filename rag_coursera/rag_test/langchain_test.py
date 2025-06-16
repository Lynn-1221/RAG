from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

def demosimple():
    try:
        prompt = ChatPromptTemplate.from_template("请介绍一下 {country}?")
        model = ChatOpenAI(
            base_url="https://xiaoai.plus/v1",
            api_key="sk-OdOQ1AkvsneaOMVVU0SaVXcqLa7oz8NZCQZ0p99iC26Uylz1",
            model_name="gpt-4o",  # 确保使用正确的模型名称
            temperature=0.5
        )

        chain = prompt | model
        response = chain.invoke({"country": "中国"}).content
        
        # 打印整个 response，看看它的结构是什么
        print(response)  # 打印 response，检查是否为字典或其他类型
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    demosimple()
