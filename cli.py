from service import Service
from dotenv import load_dotenv

load_dotenv()

def main():
    service = Service()
    print("医疗问答系统已启动（输入exit退出）")
    while True:
        query = input("用户提问: ")
        if query.lower() == 'exit':
            break
        response = service.answer(query, history=[])
        print(f"AI回答: {response}")

if __name__ == '__main__':
    main()