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

        Refresh Script App: Upload keywords and output refresh cadences based on competitor's freshness signals 

        Control Group Creator: Create correlative control groups for SEO page group testing 

        Amazon Sentiment Script: Determine sentiment of products on Amazon 

        Reddit Sentiment Script: Determine sentiment of products on reddit

        Maintained by @ccasazza
        
    """
    )

if __name__ == "__main__":
    main()