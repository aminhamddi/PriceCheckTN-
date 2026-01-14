"""\
Pipeline Monitoring Dashboard using Prefect API\
"""

import streamlit as st
from prefect.client import get_client
from datetime import datetime, timedelta
import pandas as pd

def main():
    """Main dashboard function"""
    st.title(" PriceCheckTN MLOps Pipeline Dashboard")
    st.markdown("Monitor Prefect pipeline runs and status")

    # Connect to Prefect API
    client = get_client()

    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=7))
    with col2:
        end_date = st.date_input("End Date", datetime.now())

    # Convert to datetime
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time())

    # Get flow runs
    try:
        flow_runs = client.read_flow_runs(
            flow_filter={
                "name": {"like": "PriceCheckTN MLOps Pipeline%"}
            },
            start_time=start_dt,
            end_time=end_dt
        )

        if not flow_runs:
            st.info("No pipeline runs found in the selected date range")
            return

        # Convert to DataFrame
        runs_data = []
        for run in flow_runs:
            runs_data.append({
                "run_id": run.id,
                "name": run.name,
                "state": run.state.name,
                "start_time": run.start_time,
                "end_time": run.end_time,
                "duration": (run.end_time - run.start_time).total_seconds() if run.end_time else None
            })

        df = pd.DataFrame(runs_data)

        # Display metrics
        st.subheader(" Pipeline Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Runs", len(df))
        with col2:
            success_count = len(df[df['state'] == 'Completed'])
            st.metric("Successful", success_count)
        with col3:
            failed_count = len(df[df['state'] == 'Failed'])
            st.metric("Failed", failed_count)
        with col4:
            success_rate = (success_count / len(df)) * 100 if len(df) > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")

        # Display runs table
        st.subheader(" Recent Pipeline Runs")
        st.dataframe(
            df[['run_id', 'name', 'state', 'start_time', 'end_time', 'duration']],
            use_container_width=True
        )

        # State distribution chart
        st.subheader(" Run Status Distribution")
        state_counts = df['state'].value_counts()
        st.bar_chart(state_counts)

    except Exception as e:
        st.error(f" Failed to fetch pipeline runs: {e}")

if __name__ == "__main__":
    main()