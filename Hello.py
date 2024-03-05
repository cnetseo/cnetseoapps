import streamlit as st

def main():
    st.set_page_config(
        page_title="Start Here",
        page_icon="👋",
    )

    st.write("# Welcome to RV SEO Apps 👋")

    st.sidebar.success("Choose between the following apps.")

    st.markdown(
        """
        This streamlit is a repository of data analytics and AI scripts developed by SEOs at RV. Choose from the following apps. 

        Maintained by @ccasazza
        
    """
    )

if __name__ == "__main__":
    main()