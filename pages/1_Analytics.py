import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from core.metrics import metrics_collector
from core.cache import query_cache
from datetime import datetime, timedelta

from core.vector_db import vector_db

st.set_page_config(page_title="Analytics Dashboard", layout="wide")

st.title("Fact Checker Analytics")

# Get metrics
stats = metrics_collector.get_summary()
cache_stats = query_cache.get_stats()

if stats.get('total_queries', 0) == 0:
    st.info("No data yet. Start verifying claims to see analytics!")
    st.stop()

# Overview metrics
st.subheader("Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Queries", stats['total_queries'])
with col2:
    st.metric("Avg Response Time", f"{stats.get('avg_total_time', 0):.2f}s")
with col3:
    st.metric("Avg Confidence", f"{stats.get('avg_confidence', 0):.2f}")
with col4:
    st.metric("Cache Hit Rate", f"{stats.get('cache_hit_rate', 0)*100:.1f}%")

st.divider()

# Verdict Distribution
st.subheader("Verdict Distribution")
dist = stats.get('verdict_distribution', {})

fig_pie = go.Figure(data=[go.Pie(
    labels=list(dist.keys()),
    values=list(dist.values()),
    marker=dict(colors=['#00cc00', '#cc0000', '#ffcc00'])
)])
fig_pie.update_layout(height=400)
st.plotly_chart(fig_pie, use_container_width=True)

# Performance over time
st.subheader("Performance Trends")

if len(metrics_collector.metrics) > 0:
    df = pd.DataFrame([vars(m) for m in metrics_collector.metrics[-100:]])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    fig_time = px.line(
        df,
        x='timestamp',
        y='total_time',
        title='Response Time Over Last 100 Queries'
    )
    st.plotly_chart(fig_time, use_container_width=True)
    
    # Confidence distribution
    fig_conf = px.histogram(
        df,
        x='confidence',
        nbins=20,
        title='Confidence Score Distribution'
    )
    st.plotly_chart(fig_conf, use_container_width=True)

st.subheader("Database Info")
db_stats = vector_db.get_stats()
if db_stats.get('status') == 'loaded':
    st.metric("Facts Indexed", db_stats.get('total_facts', 0))
    st.metric("Embedding Dim", db_stats.get('embedding_dim', 0))
    

# Cache statistics
st.subheader("Cache Performance")
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.metric("Cache Size", f"{cache_stats['size']}/{cache_stats['max_size']}")
with col_b:
    st.metric("TTL", f"{cache_stats['ttl_seconds']/3600:.1f}h")
with col_c:
    if st.button("Clear Cache"):
        query_cache.clear()
        st.success("Cache cleared!")
        st.rerun()

