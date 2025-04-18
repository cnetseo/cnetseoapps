import streamlit as st
import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
from datetime import datetime
import difflib
import re
import hashlib
import uuid
from collections import defaultdict

@st.cache_data
def get_wayback_content_fast(url, timestamp, timeout=20):
    """Retrieve HTML content from Wayback Machine with minimal processing"""
    try:
        wayback_url = f"http://web.archive.org/web/{timestamp}/{url}"
        response = requests.get(wayback_url, timeout=timeout)
        if response.status_code == 200:
            return response.text  # Return raw HTML without parsing
        return None
    except Exception as e:
        st.warning(f"Error retrieving {wayback_url}: {str(e)}")
        return None

@st.cache_data
def parse_date(date_string):
    formats = ["%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%Y%m%d"]
    for fmt in formats:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unable to parse date: {date_string}")

def extract_key_structures(html):
    """Extract key structures from HTML for comparison"""
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove scripts and styles
    for tag in soup(['script', 'style']):
        tag.decompose()
    
    # Extract navigation structures
    navigation = []
    nav_elements = soup.find_all(['nav', 'header', 'menu']) + soup.find_all(class_=re.compile(r'nav|menu|header', re.I))
    
    for nav in nav_elements:
        # Try to identify this nav section
        nav_id = nav.get('id', '')
        nav_class = ' '.join(nav.get('class', []))
        nav_identifier = f"{nav.name}#{nav_id}.{nav_class}"
        
        # Extract link structure
        links = []
        for a in nav.find_all('a'):
            href = a.get('href', '')
            text = a.get_text(strip=True)
            links.append(f"{text} -> {href}")
            
        navigation.append({
            'identifier': nav_identifier,
            'links': links
        })
    
    # Extract form structures
    forms = []
    for form in soup.find_all('form'):
        form_id = form.get('id', '')
        form_action = form.get('action', '')
        form_method = form.get('method', 'get')
        
        # Get form inputs
        inputs = []
        for inp in form.find_all(['input', 'select', 'textarea']):
            inp_type = inp.get('type', inp.name)
            inp_name = inp.get('name', '')
            inputs.append(f"{inp_type}:{inp_name}")
            
        forms.append({
            'identifier': f"form#{form_id}[{form_method}:{form_action}]",
            'inputs': inputs
        })
    
    # Extract iframe sources
    iframes = []
    for iframe in soup.find_all('iframe'):
        src = iframe.get('src', '')
        if src:
            iframes.append(src)
    
    # Extract layout structure (simplification of the DOM tree)
    layout = []
    for section in soup.find_all(['main', 'section', 'article', 'div']) + soup.find_all(class_=re.compile(r'container|section|content', re.I)):
        if len(section.find_all()) > 5:  # Only include substantial sections
            section_id = section.get('id', '')
            section_class = ' '.join(section.get('class', []))
            
            # Create a simplified structure representation
            children = []
            for child in section.find_all(recursive=False):
                child_class = ' '.join(child.get('class', []))
                child_id = child.get('id', '')
                children.append(f"{child.name}#{child_id}.{child_class}")
            
            layout.append({
                'identifier': f"{section.name}#{section_id}.{section_class}",
                'children': children[:10]  # Limit to first 10 children for brevity
            })
    
    return {
        'navigation': navigation,
        'forms': forms,
        'iframes': iframes,
        'layout': layout[:20]  # Limit to top 20 sections
    }

def compare_structures(structure_a, structure_b):
    """Compare key structures between two HTML versions"""
    results = {}
    
    # Compare navigation
    nav_a_ids = {nav['identifier'] for nav in structure_a['navigation']}
    nav_b_ids = {nav['identifier'] for nav in structure_b['navigation']}
    
    results['navigation'] = {
        'added': [nav for nav in structure_b['navigation'] if nav['identifier'] not in nav_a_ids],
        'removed': [nav for nav in structure_a['navigation'] if nav['identifier'] not in nav_b_ids],
        'changed': []
    }
    
    # Find changed nav sections
    for nav_a in structure_a['navigation']:
        for nav_b in structure_b['navigation']:
            if nav_a['identifier'] == nav_b['identifier']:
                # Compare links
                links_a = set(nav_a['links'])
                links_b = set(nav_b['links'])
                
                added_links = links_b - links_a
                removed_links = links_a - links_b
                
                if added_links or removed_links:
                    results['navigation']['changed'].append({
                        'identifier': nav_a['identifier'],
                        'added_links': list(added_links),
                        'removed_links': list(removed_links)
                    })
    
    # Compare forms
    form_a_ids = {form['identifier'] for form in structure_a['forms']}
    form_b_ids = {form['identifier'] for form in structure_b['forms']}
    
    results['forms'] = {
        'added': [form for form in structure_b['forms'] if form['identifier'] not in form_a_ids],
        'removed': [form for form in structure_a['forms'] if form['identifier'] not in form_b_ids],
        'changed': []
    }
    
    # Find changed forms
    for form_a in structure_a['forms']:
        for form_b in structure_b['forms']:
            if form_a['identifier'] == form_b['identifier']:
                # Compare inputs
                inputs_a = set(form_a['inputs'])
                inputs_b = set(form_b['inputs'])
                
                added_inputs = inputs_b - inputs_a
                removed_inputs = inputs_a - inputs_b
                
                if added_inputs or removed_inputs:
                    results['forms']['changed'].append({
                        'identifier': form_a['identifier'],
                        'added_inputs': list(added_inputs),
                        'removed_inputs': list(removed_inputs)
                    })
    
    # Compare iframes
    iframes_a = set(structure_a['iframes'])
    iframes_b = set(structure_b['iframes'])
    
    results['iframes'] = {
        'added': list(iframes_b - iframes_a),
        'removed': list(iframes_a - iframes_b)
    }
    
    # Compare layout
    layout_a_ids = {section['identifier'] for section in structure_a['layout']}
    layout_b_ids = {section['identifier'] for section in structure_b['layout']}
    
    results['layout'] = {
        'added': [section for section in structure_b['layout'] if section['identifier'] not in layout_a_ids],
        'removed': [section for section in structure_a['layout'] if section['identifier'] not in layout_b_ids],
        'changed': []
    }
    
    # Find changed layout sections
    for section_a in structure_a['layout']:
        for section_b in structure_b['layout']:
            if section_a['identifier'] == section_b['identifier']:
                # Compare children
                children_a = set(section_a['children'])
                children_b = set(section_b['children'])
                
                added_children = children_b - children_a
                removed_children = children_a - children_b
                
                if added_children or removed_children:
                    results['layout']['changed'].append({
                        'identifier': section_a['identifier'],
                        'added_children': list(added_children),
                        'removed_children': list(removed_children)
                    })
    
    return results

def find_most_changed_elements(html_a, html_b):
    """Identify specific elements that have changed significantly"""
    soup_a = BeautifulSoup(html_a, 'html.parser')
    soup_b = BeautifulSoup(html_b, 'html.parser')
    
    # Extract important elements with IDs
    def get_elements_with_ids(soup):
        elements = {}
        for elem in soup.find_all(id=True):
            elem_id = elem.get('id')
            # Clean the element (remove scripts, comments, etc.)
            for script in elem.find_all(['script', 'noscript', 'style']):
                script.decompose()
            
            # Store the element's HTML
            elements[elem_id] = str(elem)
        return elements
    
    elements_a = get_elements_with_ids(soup_a)
    elements_b = get_elements_with_ids(soup_b)
    
    # Find common elements
    common_ids = set(elements_a.keys()) & set(elements_b.keys())
    
    # Compare common elements and find the most changed ones
    changes = []
    
    for elem_id in common_ids:
        html_elem_a = elements_a[elem_id]
        html_elem_b = elements_b[elem_id]
        
        # Calculate similarity
        similarity = difflib.SequenceMatcher(None, html_elem_a, html_elem_b).ratio()
        
        if similarity < 0.95:  # Show elements with less than 95% similarity
            changes.append({
                'id': elem_id,
                'similarity': round(similarity * 100, 2),
                'old_content': html_elem_a[:200] + '...' if len(html_elem_a) > 200 else html_elem_a,
                'new_content': html_elem_b[:200] + '...' if len(html_elem_b) > 200 else html_elem_b
            })
    
    # Sort by most changed (lowest similarity)
    return sorted(changes, key=lambda x: x['similarity'])

def extract_selector_paths(html_content, max_paths=50):
    """Extract CSS selector paths for major elements"""
    soup = BeautifulSoup(html_content, 'html.parser')
    paths = []
    
    # Process significant elements
    for tag in soup.find_all(['div', 'section', 'article', 'nav', 'header', 'footer', 'form']):
        # Skip elements that are too small or insignificant
        if len(str(tag)) < 100 and not tag.get('id') and not tag.get('class'):
            continue
            
        # Build a selector path
        path_parts = []
        if tag.get('id'):
            path_parts.append(f"#{tag['id']}")
        elif tag.get('class'):
            class_str = '.'.join(tag['class'])
            path_parts.append(f".{class_str}")
        else:
            # For elements without ID/class, add tag and some positional info
            path_parts.append(tag.name)
            
            # Add parent context if available
            parent = tag.parent
            if parent and parent.name != '[document]':
                if parent.get('id'):
                    path_parts.insert(0, f"#{parent['id']} > ")
                elif parent.get('class'):
                    class_str = '.'.join(parent['class'])
                    path_parts.insert(0, f".{class_str} > ")
        
        selector = ''.join(path_parts)
        
        # Count children to estimate importance
        child_count = len(tag.find_all())
        
        paths.append({
            'selector': selector,
            'tag': tag.name,
            'child_count': child_count,
            'text_sample': tag.get_text()[:50] + ('...' if len(tag.get_text()) > 50 else '')
        })
    
    # Sort by number of children (most complex first)
    return sorted(paths, key=lambda x: x['child_count'], reverse=True)[:max_paths]

def compare_selector_coverage(selectors_a, selectors_b):
    """Compare selector coverage between two versions"""
    # Compare selectors by path
    selectors_a_set = {s['selector'] for s in selectors_a}
    selectors_b_set = {s['selector'] for s in selectors_b}
    
    added_selectors = selectors_b_set - selectors_a_set
    removed_selectors = selectors_a_set - selectors_b_set
    
    return {
        'added_selectors': [s for s in selectors_b if s['selector'] in added_selectors],
        'removed_selectors': [s for s in selectors_a if s['selector'] in removed_selectors]
    }

def detailed_html_comparison(html_a, html_b):
    """Perform detailed HTML comparison"""
    # Extract key structures
    structures_a = extract_key_structures(html_a)
    structures_b = extract_key_structures(html_b)
    
    # Compare structures
    structure_diff = compare_structures(structures_a, structures_b)
    
    # Find most changed elements by ID
    changed_elements = find_most_changed_elements(html_a, html_b)
    
    # Extract selector paths
    selectors_a = extract_selector_paths(html_a)
    selectors_b = extract_selector_paths(html_b)
    
    # Compare selector coverage
    selector_diff = compare_selector_coverage(selectors_a, selectors_b)
    
    # Return comprehensive analysis
    return {
        'structure_diff': structure_diff,
        'changed_elements': changed_elements,
        'selector_diff': selector_diff
    }

def compare_wayback_html_detailed(url, date1_str, date2_str):
    """Detailed comparison of HTML between two versions"""
    date1 = parse_date(date1_str)
    date2 = parse_date(date2_str)

    timestamp1 = date1.strftime("%Y%m%d")
    timestamp2 = date2.strftime("%Y%m%d")

    progress_text = "Retrieving and analyzing HTML..."
    my_bar = st.progress(0, text=progress_text)
    
    try:
        with st.spinner(f"Retrieving HTML from {date1.date()}..."):
            html_a = get_wayback_content_fast(url, timestamp1)
            my_bar.progress(0.25)
        
        with st.spinner(f"Retrieving HTML from {date2.date()}..."):
            html_b = get_wayback_content_fast(url, timestamp2)
            my_bar.progress(0.5)

        if not html_a or not html_b:
            return f"Failed to retrieve HTML for {url}", None

        with st.spinner("Performing detailed structure analysis..."):
            result = detailed_html_comparison(html_a, html_b)
            my_bar.progress(1.0)
            
        return f"HTML comparison for {url}", result
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, None
    finally:
        my_bar.empty()

def main():
    st.title("HTML Structure Diff Analyzer")
    
    url = st.text_input("Enter a URL:", "https://www.example.com")
    
    col1, col2 = st.columns(2)
    with col1:
        date1 = st.date_input("First date:")
    with col2:
        date2 = st.date_input("Second date:")

    if st.button("Analyze HTML Changes"):
        title, comparison = compare_wayback_html_detailed(
            url, 
            date1.strftime("%Y-%m-%d"), 
            date2.strftime("%Y-%m-%d")
        )
        
        if comparison:
            st.write(title)
            
            # Create tabs for different types of changes
            tab1, tab2, tab3, tab4 = st.tabs(["Navigation Changes", "Layout Changes", "Form Changes", "Element Changes"])
            
            with tab1:
                st.subheader("Navigation Changes")
                
                nav_changes = comparison['structure_diff']['navigation']
                
                if nav_changes['added']:
                    st.write("🟢 **Added Navigation Sections:**")
                    for nav in nav_changes['added']:
                        st.write(f"- {nav['identifier']}")
                        if nav['links']:
                            with st.expander(f"View {len(nav['links'])} links"):
                                for link in nav['links']:
                                    st.write(f"  • {link}")
                
                if nav_changes['removed']:
                    st.write("🔴 **Removed Navigation Sections:**")
                    for nav in nav_changes['removed']:
                        st.write(f"- {nav['identifier']}")
                        if nav['links']:
                            with st.expander(f"View {len(nav['links'])} links"):
                                for link in nav['links']:
                                    st.write(f"  • {link}")
                
                if nav_changes['changed']:
                    st.write("🟠 **Changed Navigation Sections:**")
                    for nav in nav_changes['changed']:
                        st.write(f"- {nav['identifier']}")
                        
                        if nav['added_links']:
                            with st.expander(f"View {len(nav['added_links'])} added links"):
                                for link in nav['added_links']:
                                    st.write(f"  ✅ {link}")
                        
                        if nav['removed_links']:
                            with st.expander(f"View {len(nav['removed_links'])} removed links"):
                                for link in nav['removed_links']:
                                    st.write(f"  ❌ {link}")
                
                if not nav_changes['added'] and not nav_changes['removed'] and not nav_changes['changed']:
                    st.write("No significant changes to navigation detected.")
            
            with tab2:
                st.subheader("Layout Changes")
                
                layout_changes = comparison['structure_diff']['layout']
                
                if layout_changes['added']:
                    st.write("🟢 **Added Layout Sections:**")
                    for section in layout_changes['added']:
                        st.write(f"- {section['identifier']}")
                        if section['children']:
                            with st.expander(f"View children elements"):
                                for child in section['children']:
                                    st.write(f"  • {child}")
                
                if layout_changes['removed']:
                    st.write("🔴 **Removed Layout Sections:**")
                    for section in layout_changes['removed']:
                        st.write(f"- {section['identifier']}")
                        if section['children']:
                            with st.expander(f"View children elements"):
                                for child in section['children']:
                                    st.write(f"  • {child}")
                
                if layout_changes['changed']:
                    st.write("🟠 **Changed Layout Sections:**")
                    for section in layout_changes['changed']:
                        st.write(f"- {section['identifier']}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if section['added_children']:
                                with st.expander(f"View {len(section['added_children'])} added elements"):
                                    for child in section['added_children']:
                                        st.write(f"  ✅ {child}")
                        
                        with col2:
                            if section['removed_children']:
                                with st.expander(f"View {len(section['removed_children'])} removed elements"):
                                    for child in section['removed_children']:
                                        st.write(f"  ❌ {child}")
                
                # Show selector changes
                selector_diff = comparison['selector_diff']
                
                if selector_diff['added_selectors']:
                    st.write("🟢 **Added Major Elements (by CSS selector):**")
                    for selector in selector_diff['added_selectors'][:10]:  # Limit to top 10
                        st.write(f"- `{selector['selector']}` ({selector['tag']} with {selector['child_count']} children)")
                        st.write(f"  Text sample: {selector['text_sample']}")
                
                if selector_diff['removed_selectors']:
                    st.write("🔴 **Removed Major Elements (by CSS selector):**")
                    for selector in selector_diff['removed_selectors'][:10]:  # Limit to top 10
                        st.write(f"- `{selector['selector']}` ({selector['tag']} with {selector['child_count']} children)")
                        st.write(f"  Text sample: {selector['text_sample']}")
                
                if not layout_changes['added'] and not layout_changes['removed'] and not layout_changes['changed'] and not selector_diff['added_selectors'] and not selector_diff['removed_selectors']:
                    st.write("No significant changes to layout detected.")
            
            with tab3:
                st.subheader("Form Changes")
                
                form_changes = comparison['structure_diff']['forms']
                
                if form_changes['added']:
                    st.write("🟢 **Added Forms:**")
                    for form in form_changes['added']:
                        st.write(f"- {form['identifier']}")
                        if form['inputs']:
                            with st.expander(f"View {len(form['inputs'])} inputs"):
                                for input in form['inputs']:
                                    st.write(f"  • {input}")
                
                if form_changes['removed']:
                    st.write("🔴 **Removed Forms:**")
                    for form in form_changes['removed']:
                        st.write(f"- {form['identifier']}")
                        if form['inputs']:
                            with st.expander(f"View {len(form['inputs'])} inputs"):
                                for input in form['inputs']:
                                    st.write(f"  • {input}")
                
                if form_changes['changed']:
                    st.write("🟠 **Changed Forms:**")
                    for form in form_changes['changed']:
                        st.write(f"- {form['identifier']}")
                        
                        if form['added_inputs']:
                            with st.expander(f"View {len(form['added_inputs'])} added inputs"):
                                for input in form['added_inputs']:
                                    st.write(f"  ✅ {input}")
                        
                        if form['removed_inputs']:
                            with st.expander(f"View {len(form['removed_inputs'])} removed inputs"):
                                for input in form['removed_inputs']:
                                    st.write(f"  ❌ {input}")
                
                # Show iframe changes
                iframe_changes = comparison['structure_diff']['iframes']
                
                if iframe_changes['added']:
                    st.write("🟢 **Added iFrames:**")
                    for src in iframe_changes['added']:
                        st.write(f"- `{src}`")
                
                if iframe_changes['removed']:
                    st.write("🔴 **Removed iFrames:**")
                    for src in iframe_changes['removed']:
                        st.write(f"- `{src}`")
                
                if not form_changes['added'] and not form_changes['removed'] and not form_changes['changed'] and not iframe_changes['added'] and not iframe_changes['removed']:
                    st.write("No significant changes to forms or iframes detected.")
            
            with tab4:
                st.subheader("Changed Elements by ID")
                
                changed_elements = comparison['changed_elements']
                
                if changed_elements:
                    st.write(f"Found {len(changed_elements)} elements with significant changes:")
                    
                    for i, elem in enumerate(changed_elements[:10]):  # Limit to top 10
                        st.write(f"**{i+1}. Element ID: `{elem['id']}` (Similarity: {elem['similarity']}%)**")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("Before:")
                            st.code(elem['old_content'], language='html')
                        
                        with col2:
                            st.write("After:")
                            st.code(elem['new_content'], language='html')
                else:
                    st.write("No significant changes to specific elements detected.")
                
        else:
            st.error("Failed to compare HTML. Please try again.")

if __name__ == "__main__":
    main()