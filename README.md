# DAPartner

**DAPartner** is an open-source tool for modeling the **star schema** layer in data warehouses.  
It enables requirements gathering, creation of dimensions and fact tables following the **Kimball** approach, and then supports the process of documenting and automating DWH projects.

## 🔧 Tech Stack

The project uses:
- [Streamlit](https://streamlit.io/) – user interface  
- [LangChain](https://www.langchain.com/) / [LangGraph](https://github.com/langchain-ai/langgraph) – agent logic  
- [Pandas](https://pandas.pydata.org/) – data operations  
- [Pydantic](https://docs.pydantic.dev/) – structure validation  
- [sqlparse](https://github.com/andialbrecht/sqlparse) – SQL parsing  

---

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/rajmundkulawinek/DAPartner.git
cd DAPartner
