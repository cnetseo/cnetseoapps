import streamlit as st
import pandas as pd

def ensure_supabase():
    try:
        import supabase
        return True
    except ImportError:
        import subprocess
        import sys
        
        print("Installing Supabase dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "supabase", "websockets==12.0", "--quiet"
        ])
        
        try:
            import supabase
            return True
        except ImportError:
            st.error("Failed to install Supabase. Please install it manually.")
            return False

# Call this at the start of your script
if not ensure_supabase():
    st.stop() 
    
from supabase import create_client

# Initialize Supabase client
@st.cache_resource
def init_supabase():
    url = st.secrets['SUPABASE']["SUPABASE_URL"]
    key = st.secrets['SUPABASE']["SUPABASE_KEY"]
    return create_client(url, key)

def load_categories():
    """Load unique promo_gl values from Supabase"""
    try:
        supabase = init_supabase()
        response = supabase.rpc(
            'get_unique_promo_gl',
            {}
        ).execute()
        
        if response.data:
            # Extract just the category value from each object
            return [item['category'] for item in response.data]
        return []
    except Exception as e:
        st.error(f"Error loading promo_gl values: {str(e)}")
        return []
    
def load_brands():
    """Load unique brands from Supabase"""
    try:
        supabase = init_supabase()
        response = supabase.rpc(
            'get_unique_brands',
            {}
        ).execute()
        
        if response.data:
            return [item['brand'] for item in response.data]
        return []
    except Exception as e:
        st.error(f"Error loading brands: {str(e)}")
        return []

def filter_deals(filters):
    """
    Filter deals using Supabase queries
    
    Parameters:
    - filters: Dictionary containing filter parameters
    """
    try:
        supabase = init_supabase()
        
        # Base query
        query = supabase.table("amazon_spring_sale").select("*")
        
        # Apply filters
        if filters.get('categories'):
            query = query.in_('promo_gl', filters['categories'])
            
        if filters.get('brand') and filters['brand'] != 'All':
            query = query.eq('brand', filters['brand'])
            
        if filters.get('min_rating'):
            query = query.gte('star_rating', filters['min_rating'])
            
        if filters.get('min_discount'):
            query = query.gte('discount_percentage', filters['min_discount'])
            
        if filters.get('max_price'):
            query = query.lte('deal_price', filters['max_price'])
            
        if filters.get('max_vs_ytd'):
            # This requires a custom SQL function in Supabase
            query = query.lte('ytd_price_difference_pct', filters['max_vs_ytd'])
            
        # Add ordering - using the latest Supabase syntax
        if filters.get('order_by'):
            query = query.order(filters['order_by'], desc=filters.get('order_desc', False))
        
        # Add pagination
        offset = filters.get('offset', 0)
        limit = filters.get('limit', 100)
        query = query.range(offset, offset + limit - 1)
        
        # Execute the query
        response = query.execute()
        
        # Get total count (requires a separate query)
        count_query = supabase.table("amazon_spring_sale").select("*", count='exact')
        
        # Apply same filters to count query
        if filters.get('categories'):
            count_query = count_query.in_('promo_gl', filters['categories'])
            
        if filters.get('brand') and filters['brand'] != 'All':
            count_query = count_query.eq('brand', filters['brand'])
            
        if filters.get('min_rating'):
            count_query = count_query.gte('star_rating', filters['min_rating'])
            
        if filters.get('min_discount'):
            count_query = count_query.gte('discount_percentage', filters['min_discount'])
            
        if filters.get('max_price'):
            count_query = count_query.lte('deal_price', filters['max_price'])
            
        if filters.get('max_vs_ytd'):
            count_query = count_query.lte('ytd_price_difference_pct', filters['max_vs_ytd'])
            
        count_response = count_query.execute()
        total_count = count_response.count
        
        return response.data, total_count
        
    except Exception as e:
        st.error(f"Error filtering deals: {str(e)}")
        st.write("Full error details:", str(e))
        return [], 0

def format_deal_display(df):
    """Format dataframe for display"""
    if df.empty:
        return df
        
    display_df = df.copy()
    
    # Format metrics
    if 'discount_percentage' in display_df.columns:
        display_df['discount_percentage'] = display_df['discount_percentage'].apply(
            lambda x: f"{float(x):.1f}%" if pd.notna(x) else "N/A"
        )
    
    if 'deal_price' in display_df.columns:
        display_df['deal_price'] = display_df['deal_price'].apply(
            lambda x: f"${float(x):.2f}" if pd.notna(x) else "N/A"
        )
    
    if 'star_rating' in display_df.columns:
        display_df['star_rating'] = display_df['star_rating'].apply(
            lambda x: f"{float(x):.1f}⭐" if pd.notna(x) else "N/A"
        )
    
    if 'lowest_price_ytd' in display_df.columns:
        display_df['lowest_price_ytd'] = display_df['lowest_price_ytd'].apply(
            lambda x: f"${float(x):.2f}" if pd.notna(x) else "N/A"
        )
    
    return display_df

def main():
    st.title("Amazon Spring Sale Filter")
    
    # Initialize session state for filters if it doesn't exist
    if 'filters' not in st.session_state:
        st.session_state.filters = {
            'categories': [],
            'brand': 'All',
            'min_rating': 0.0,
            'min_discount': 0.0,
            'max_price': 1000.0,
            'max_vs_ytd': 100,
            'order_by': 'deal_price',
            'order_desc': False,
            'offset': 0,
            'limit': 100
        }
    
    # Sidebar for filters
    st.sidebar.header("Filter Options")
    
    # Load categories and brands from Supabase
    all_categories = load_categories()
    all_brands = ['All'] + load_brands()
    
    # Filter out any saved categories that no longer exist in the available options
    valid_categories = [cat for cat in st.session_state.filters['categories'] if cat in all_categories]
    
    # Category selection with session state
    st.session_state.filters['categories'] = st.sidebar.multiselect(
        "Select Categories",
        all_categories,
        default=valid_categories,
        help="Select categories to include in results"
    )
    
    # Brand filter in sidebar with session state
    brand_index = 0
    try:
        if st.session_state.filters['brand'] in all_brands:
            brand_index = all_brands.index(st.session_state.filters['brand'])
    except (ValueError, Exception):
        pass
    
    st.session_state.filters['brand'] = st.sidebar.selectbox(
        "Brand",
        all_brands,
        index=brand_index
    )
    
    # Numeric filters
    st.sidebar.subheader("Deal Metrics")
    
    st.session_state.filters['min_rating'] = st.sidebar.slider(
        "Min Rating", 
        0.0, 5.0, 
        st.session_state.filters['min_rating'], 
        0.1
    )
    
    st.session_state.filters['min_discount'] = st.sidebar.slider(
        "Min Discount %", 
        0.0, 100.0, 
        st.session_state.filters['min_discount'], 
        5.0
    )
    
    st.session_state.filters['max_price'] = st.sidebar.slider(
        "Max Price ($)", 
        0.0, 1000.0, 
        st.session_state.filters['max_price'], 
        10.0
    )
    
    st.session_state.filters['max_vs_ytd'] = st.sidebar.slider(
        "Max % Above YTD Low", 
        0, 100, 
        st.session_state.filters['max_vs_ytd']
    )
    
    # Sorting options
    st.sidebar.subheader("Sort Results")
    
    sort_options = {
        'deal_price': 'Price',
        'discount_percentage': 'Discount %',
        'star_rating': 'Rating',
        'ytd_price_difference_pct': 'YTD Price Difference %'
    }
    
    sort_index = 0
    try:
        if st.session_state.filters['order_by'] in sort_options:
            sort_index = list(sort_options.keys()).index(st.session_state.filters['order_by'])
    except (ValueError, Exception):
        pass
    
    st.session_state.filters['order_by'] = st.sidebar.selectbox(
        "Sort By",
        options=list(sort_options.keys()),
        format_func=lambda x: sort_options[x],
        index=sort_index
    )
    
    st.session_state.filters['order_desc'] = st.sidebar.checkbox(
        "Descending Order",
        value=st.session_state.filters['order_desc']
    )
    
    # Main content
    if st.button("Apply Filters"):
        with st.spinner('Filtering deals...'):
            # Get deals from Supabase with filters
            filtered_data, total_count = filter_deals(st.session_state.filters)
            
            if filtered_data:
                filtered_df = pd.DataFrame(filtered_data)
                display_df = format_deal_display(filtered_df)
                
                # Show results
                st.subheader(f"Filtered Deals ({total_count} total)")
                
                # Determine which columns to display based on what's available
                display_columns = ['asin_name', 'brand', 'deal_price', 
                                  'discount_percentage', 'star_rating', 
                                  'lowest_price_ytd', 'promo_gl']
                
                # Filter to only include columns that exist in the dataframe
                actual_columns = [col for col in display_columns if col in display_df.columns]
                
                st.dataframe(display_df[actual_columns])
                
                # Pagination controls
                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    if st.session_state.filters['offset'] > 0:
                        if st.button("Previous Page"):
                            st.session_state.filters['offset'] = max(
                                0, 
                                st.session_state.filters['offset'] - st.session_state.filters['limit']
                            )
                            st.rerun()
                
                with col2:
                    current_page = st.session_state.filters['offset'] // st.session_state.filters['limit'] + 1
                    total_pages = (total_count - 1) // st.session_state.filters['limit'] + 1
                    st.write(f"Page {current_page} of {total_pages}")
                
                with col3:
                    if st.session_state.filters['offset'] + st.session_state.filters['limit'] < total_count:
                        if st.button("Next Page"):
                            st.session_state.filters['offset'] += st.session_state.filters['limit']
                            st.rerun()
                
                # Download option
                st.download_button(
                    "Download Results",
                    filtered_df.to_csv(index=False),
                    "filtered_deals.csv",
                    "text/csv",
                    key='download-results'
                )
                
            else:
                st.warning("No deals match your filters. Try adjusting your criteria.")
    
    # Instructions
    with st.expander("How to use this tool"):
        st.write("""
        1. Select filter criteria in the sidebar
        2. Click 'Apply Filters' to search for deals
        3. Browse results in the table
        4. Use pagination controls to navigate through results
        5. Download filtered data as CSV for further analysis
        """)

if __name__ == "__main__":
    main()