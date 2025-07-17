import streamlit as st
import pandas as pd
import numpy as np
import pingouin as pg
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# Set page configuration
st.set_page_config(
	page_title="Mediation Analysis Explained",
	page_icon="📊",
	layout="wide",
	initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .title {
        font-size: 42px;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 20px;
    }
    .subtitle {
        font-size: 26px;
        font-weight: bold;
        color: #43A047;
        margin-top: 30px;
        margin-bottom: 15px;
    }
    .section {
        font-size: 22px;
        font-weight: bold;
        color: #FFA000;
        margin-top: 25px;
        margin-bottom: 10px;
    }
    .highlight {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #1E88E5;
        margin-bottom: 20px;
    }
    .author {
        font-size: 18px;
        font-style: italic;
        color: #616161;
        text-align: center;
        margin-bottom: 30px;
    }
    .formula-box {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #e0e0e0;
        margin: 10px 0;
    }
    .tip {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #43A047;
        margin-bottom: 20px;
    }
    .warning {
        background-color: #fff8e1;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #FFA000;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title">Understanding Mediation Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="author">By Dr. Merwan Roudane</div>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
	"Go to",
	["Introduction", "Basic Concepts", "Types of Mediation", "Statistical Models",
	 "Implementation with Pingouin", "Interactive Example", "Interpretation", "References"]
)


# Generate example data for demonstration
@st.cache_data
def generate_example_data(n=200, seed=42):
	np.random.seed(seed)

	# Generate predictor variable X
	X = np.random.normal(0, 1, n)

	# Generate mediator M with effect from X
	M = 0.5 * X + np.random.normal(0, 1, n)

	# Generate outcome Y with effects from both X and M
	Y = 0.3 * X + 0.6 * M + np.random.normal(0, 1, n)

	# Create a binary mediator as well
	Mbin = (M > np.median(M)).astype(int)

	# Generate a binary outcome
	Ybin = (Y > np.median(Y)).astype(int)

	# Combine into a dataframe
	df = pd.DataFrame({
		'X': X,
		'M': M,
		'Mbin': Mbin,
		'Y': Y,
		'Ybin': Ybin
	})

	return df


# Generate data
example_data = generate_example_data()


# Function to create basic mediation diagram
def create_mediation_diagram(indirect_effect=0.3, direct_effect=0.2, xm_effect=0.5, my_effect=0.6):
	fig = go.Figure()

	# Add nodes
	fig.add_trace(go.Scatter(
		x=[0, 2, 4],
		y=[0, 1, 0],
		mode='markers+text',
		marker=dict(size=40, color=['#1E88E5', '#43A047', '#FFA000']),
		text=['X', 'M', 'Y'],
		textfont=dict(size=20, color='white'),
		hoverinfo='skip'
	))

	# Add arrows
	# Direct effect (X -> Y)
	fig.add_annotation(
		x=0, y=0,
		ax=4, ay=0,
		xref="x", yref="y",
		axref="x", ayref="y",
		showarrow=True,
		arrowhead=2,
		arrowsize=1.5,
		arrowwidth=2,
		arrowcolor="#FF5722",
		text=f"c' = {direct_effect}",
		font=dict(size=14, color="#FF5722"),
		textangle=0,
		align="center",
		opacity=0.8
	)

	# X -> M
	fig.add_annotation(
		x=0, y=0,
		ax=2, ay=1,
		xref="x", yref="y",
		axref="x", ayref="y",
		showarrow=True,
		arrowhead=2,
		arrowsize=1.5,
		arrowwidth=2,
		arrowcolor="#2196F3",
		text=f"a = {xm_effect}",
		font=dict(size=14, color="#2196F3"),
		textangle=30,
		align="center",
		opacity=0.8
	)

	# M -> Y
	fig.add_annotation(
		x=2, y=1,
		ax=4, ay=0,
		xref="x", yref="y",
		axref="x", ayref="y",
		showarrow=True,
		arrowhead=2,
		arrowsize=1.5,
		arrowwidth=2,
		arrowcolor="#4CAF50",
		text=f"b = {my_effect}",
		font=dict(size=14, color="#4CAF50"),
		textangle=-30,
		align="center",
		opacity=0.8
	)

	# Add indirect effect label
	fig.add_annotation(
		x=2, y=0.5,
		text=f"Indirect effect: a×b = {indirect_effect}",
		showarrow=False,
		font=dict(size=14, color="#673AB7"),
		align="center",
		opacity=0.8
	)

	# Update layout
	fig.update_layout(
		width=700,
		height=400,
		xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 5]),
		yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 1.5]),
		showlegend=False,
		margin=dict(l=20, r=20, t=20, b=20),
		plot_bgcolor='rgba(0,0,0,0)'
	)

	return fig


# Function to create bootstrap distribution visualization
def create_bootstrap_plot(df):
	# Perform mediation analysis
	result, dist = pg.mediation_analysis(
		data=df,
		x='X',
		m='M',
		y='Y',
		n_boot=1000,
		return_dist=True,
		seed=42
	)

	# Create density plot
	fig = px.histogram(
		x=dist,
		nbins=30,
		histnorm='probability density',
		title='Bootstrap Distribution of Indirect Effect',
		labels={'x': 'Indirect Effect', 'y': 'Density'},
		color_discrete_sequence=['#673AB7']
	)

	# Add vertical line for the indirect effect
	indirect_effect = result.loc[result['path'] == 'Indirect', 'coef'].values[0]
	ci_lower = result.loc[result['path'] == 'Indirect', 'CI[2.5%]'].values[0]
	ci_upper = result.loc[result['path'] == 'Indirect', 'CI[97.5%]'].values[0]

	fig.add_vline(x=indirect_effect, line_width=2, line_dash="solid", line_color="#FF5722")
	fig.add_vline(x=ci_lower, line_width=2, line_dash="dash", line_color="#FFA000")
	fig.add_vline(x=ci_upper, line_width=2, line_dash="dash", line_color="#FFA000")

	# Add annotations
	fig.add_annotation(
		x=indirect_effect,
		y=0.5,
		text=f"Indirect Effect: {indirect_effect:.3f}",
		showarrow=True,
		arrowhead=1,
		ax=30,
		ay=-30,
		font=dict(color="#FF5722", size=12)
	)

	fig.add_annotation(
		x=(ci_lower + ci_upper) / 2,
		y=0.2,
		text=f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]",
		showarrow=False,
		font=dict(color="#FFA000", size=12)
	)

	# Update layout
	fig.update_layout(
		width=700,
		height=400,
		xaxis_title="Indirect Effect (a×b)",
		yaxis_title="Density",
		template="plotly_white"
	)

	return fig, result


# Function to create scatter plot for relationships
def create_relationship_plots(df):
	# Create a figure with subplots
	fig = make_subplots(rows=1, cols=3,
						subplot_titles=("X → M Relationship",
										"M → Y Relationship",
										"X → Y Relationship"))

	# X → M plot
	fig.add_trace(
		go.Scatter(
			x=df['X'],
			y=df['M'],
			mode='markers',
			marker=dict(color='#2196F3', size=8, opacity=0.7),
			name='X → M'
		),
		row=1, col=1
	)

	# Add regression line for X → M
	x_range = np.linspace(min(df['X']), max(df['X']), 100)
	model = np.polyfit(df['X'], df['M'], 1)
	y_pred = model[0] * x_range + model[1]
	fig.add_trace(
		go.Scatter(
			x=x_range,
			y=y_pred,
			mode='lines',
			line=dict(color='#1565C0', width=3),
			name='a path'
		),
		row=1, col=1
	)

	# M → Y plot
	fig.add_trace(
		go.Scatter(
			x=df['M'],
			y=df['Y'],
			mode='markers',
			marker=dict(color='#4CAF50', size=8, opacity=0.7),
			name='M → Y'
		),
		row=1, col=2
	)

	# Add regression line for M → Y
	m_range = np.linspace(min(df['M']), max(df['M']), 100)
	model = np.polyfit(df['M'], df['Y'], 1)
	y_pred = model[0] * m_range + model[1]
	fig.add_trace(
		go.Scatter(
			x=m_range,
			y=y_pred,
			mode='lines',
			line=dict(color='#2E7D32', width=3),
			name='b path'
		),
		row=1, col=2
	)

	# X → Y plot
	fig.add_trace(
		go.Scatter(
			x=df['X'],
			y=df['Y'],
			mode='markers',
			marker=dict(color='#FFA000', size=8, opacity=0.7),
			name='X → Y'
		),
		row=1, col=3
	)

	# Add regression line for X → Y
	model = np.polyfit(df['X'], df['Y'], 1)
	y_pred = model[0] * x_range + model[1]
	fig.add_trace(
		go.Scatter(
			x=x_range,
			y=y_pred,
			mode='lines',
			line=dict(color='#F57C00', width=3),
			name='c path'
		),
		row=1, col=3
	)

	# Update layout
	fig.update_layout(
		height=400,
		width=900,
		showlegend=False,
		template="plotly_white"
	)

	# Update axes
	fig.update_xaxes(title_text="Predictor (X)", row=1, col=1)
	fig.update_yaxes(title_text="Mediator (M)", row=1, col=1)
	fig.update_xaxes(title_text="Mediator (M)", row=1, col=2)
	fig.update_yaxes(title_text="Outcome (Y)", row=1, col=2)
	fig.update_xaxes(title_text="Predictor (X)", row=1, col=3)
	fig.update_yaxes(title_text="Outcome (Y)", row=1, col=3)

	return fig


# Main content based on selected page
if page == "Introduction":
	st.markdown('<div class="subtitle">What is Mediation Analysis?</div>', unsafe_allow_html=True)

	st.markdown("""
    <div class="highlight">
    Mediation analysis is a statistical approach used to understand the underlying mechanisms or processes by which one variable influences another variable. It helps researchers identify and explain the causal pathways between variables.
    </div>
    """, unsafe_allow_html=True)

	# Mediation diagram
	st.markdown('<div class="section">Basic Mediation Model</div>', unsafe_allow_html=True)
	fig = create_mediation_diagram()
	st.plotly_chart(fig, use_container_width=True)

	st.markdown("""
    In this diagram:
    - **X** is the independent variable (predictor)
    - **M** is the mediating variable (mediator)
    - **Y** is the dependent variable (outcome)
    - The **direct effect** is the direct influence of X on Y (path c')
    - The **indirect effect** is the influence of X on Y through M (path a × path b)
    - The **total effect** is the sum of direct and indirect effects (c = c' + a×b)
    """)

	st.markdown('<div class="section">Why is Mediation Analysis Important?</div>', unsafe_allow_html=True)

	col1, col2 = st.columns(2)

	with col1:
		st.markdown("""
        Mediation analysis helps researchers:

        1. **Understand mechanisms**: Reveals how and why effects occur
        2. **Test theories**: Validates theoretical frameworks about causal relationships
        3. **Design interventions**: Identifies key points for effective interventions
        4. **Explain complex relationships**: Uncovers hidden relationships between variables
        """)

	with col2:
		st.markdown("""
        <div class="tip">
        <strong>Example:</strong> Researchers studying the effect of exercise (X) on depression (Y) might hypothesize that improved sleep quality (M) mediates this relationship. Mediation analysis would help determine if exercise reduces depression directly or by first improving sleep quality, which then reduces depression.
        </div>
        """, unsafe_allow_html=True)

	st.markdown('<div class="section">Historical Context</div>', unsafe_allow_html=True)

	st.markdown("""
    The concept of mediation analysis was formalized by Baron and Kenny (1986) in their seminal paper. Their approach, often called the "causal steps approach," provided a systematic method for testing mediation. Since then, more sophisticated methods have been developed, including bootstrap methods for estimating indirect effects and their confidence intervals.
    """)

elif page == "Basic Concepts":
	st.markdown('<div class="subtitle">Fundamental Concepts in Mediation Analysis</div>', unsafe_allow_html=True)

	st.markdown('<div class="section">Key Variables and Paths</div>', unsafe_allow_html=True)

	col1, col2 = st.columns([3, 2])

	with col1:
		st.markdown("""
        In mediation analysis, we consider three main variables:

        1. **Independent Variable (X)**: The predictor or causal variable
        2. **Mediator Variable (M)**: The intervening variable that explains the relationship
        3. **Dependent Variable (Y)**: The outcome variable

        The relationships between these variables are represented by several paths:

        - **Path a**: The effect of X on M
        - **Path b**: The effect of M on Y (controlling for X)
        - **Path c**: The total effect of X on Y
        - **Path c'**: The direct effect of X on Y (controlling for M)
        """)

	with col2:
		st.latex(r'''
        \begin{align}
        M &= i_1 + aX + e_1 \\
        Y &= i_2 + c'X + bM + e_2 \\
        Y &= i_3 + cX + e_3
        \end{align}
        ''')

	st.markdown('<div class="section">Types of Effects in Mediation</div>', unsafe_allow_html=True)

	col1, col2 = st.columns(2)

	with col1:
		st.markdown("""
        **1. Direct Effect (c')**
        - The effect of X on Y that is independent of the mediator
        - Represented by the path c'

        **2. Indirect Effect (a×b)**
        - The effect of X on Y through the mediator M
        - Calculated as the product of paths a and b
        - Also called the mediated effect

        **3. Total Effect (c)**
        - The overall effect of X on Y
        - Sum of direct and indirect effects: c = c' + a×b
        """)

	with col2:
		st.markdown('<div class="formula-box">', unsafe_allow_html=True)
		st.latex(r'''
        \begin{align}
        \text{Direct Effect} &= c' \\
        \text{Indirect Effect} &= a \times b \\
        \text{Total Effect} &= c = c' + a \times b
        \end{align}
        ''')
		st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<div class="section">Assumptions of Mediation Analysis</div>', unsafe_allow_html=True)

	st.markdown("""
    <div class="warning">
    For valid mediation analysis, several key assumptions must be met:

    1. **Temporal precedence**: X must precede M, and M must precede Y
    2. **No unmeasured confounders**: No unmeasured variables should affect the relationships between X, M, and Y
    3. **Reliability of measures**: All variables should be measured with minimal error
    4. **Model specification**: The relationships between variables should be correctly specified
    5. **No reverse causality**: M should not cause X, and Y should not cause M or X
    </div>
    """, unsafe_allow_html=True)

	st.markdown('<div class="section">Conditions for Mediation</div>', unsafe_allow_html=True)

	st.markdown("""
    According to Baron and Kenny (1986), four conditions should be met to establish mediation:

    1. X significantly predicts Y (path c)
    2. X significantly predicts M (path a)
    3. M significantly predicts Y when controlling for X (path b)
    4. The effect of X on Y decreases (partial mediation) or becomes non-significant (complete mediation) when M is included in the model

    Modern approaches focus more on the significance of the indirect effect (a×b) rather than these steps.
    """)

elif page == "Types of Mediation":
	st.markdown('<div class="subtitle">Different Types of Mediation</div>', unsafe_allow_html=True)

	st.markdown('<div class="section">Complete vs. Partial Mediation</div>', unsafe_allow_html=True)

	col1, col2 = st.columns(2)

	with col1:
		# Complete mediation diagram
		st.markdown("**Complete Mediation**")
		fig_complete = create_mediation_diagram(indirect_effect=0.3, direct_effect=0.0, xm_effect=0.5, my_effect=0.6)
		st.plotly_chart(fig_complete, use_container_width=True)

	with col2:
		# Partial mediation diagram
		st.markdown("**Partial Mediation**")
		fig_partial = create_mediation_diagram(indirect_effect=0.3, direct_effect=0.2, xm_effect=0.5, my_effect=0.6)
		st.plotly_chart(fig_partial, use_container_width=True)

	st.markdown("""
    **Complete Mediation** occurs when the mediator fully explains the relationship between X and Y. In this case:
    - The direct effect (c') becomes zero or non-significant
    - The total effect (c) equals the indirect effect (a×b)

    **Partial Mediation** occurs when the mediator partially explains the relationship between X and Y. In this case:
    - Both direct effect (c') and indirect effect (a×b) are significant
    - The total effect (c) is the sum of direct and indirect effects
    """)

	st.markdown('<div class="section">Simple vs. Multiple Mediator Models</div>', unsafe_allow_html=True)

	col1, col2 = st.columns(2)

	with col1:
		st.markdown("**Simple Mediation**")
		# Use the same diagram as before
		st.image(
			"https://mermaid.ink/img/pako:eNpVj80KwjAQhF-l7KmFgmCPngRPHtRLbzWGNJrFJJJsQRR8d9OKP3OawXxfZmY4Y-IZYYi3JJhWfV6PSVTCBJ1gUg4qXhQEoaBIWaCzCJZSEQmvvC7BmQaNYzQarwiCr5UdBLMO3F5qShtDMiX9lfaXruQHiS1Nmu1QWnWvjBWRb_qNWOdunrFZYDzwluZpYuR9VK3JmT0yXrJuXA_e78mENdjwUuVrMnDC4R_wBk5-Rbk",
			width=350)

	with col2:
		st.markdown("**Multiple Parallel Mediators**")
		st.image(
			"https://mermaid.ink/img/pako:eNptkMsKwjAQRX-lzKqFgmBXXQiuXKibbmsMaTSLSSRpQRH_3fQhPmY1c-_hMJMzpolTDPGaBUurv1_nJCphgk4wKQeVLgqCUFAkLNBZBMOoiIRXwUtwpkHjGLXGG4Lga2UHwawDd0vN-WBIJ-QvdXxrOn6Q2fLRs8ylVc_KWBH5od-IdenWOZsF5pPcMz5MDL-OajQlsSdOV6HblIP3ZzJhDTa8VGVNBk44_gu-HG9N2g",
			width=350)

	st.markdown("""
    **Simple Mediation** involves a single mediator between X and Y.

    **Multiple Mediator Models** can take different forms:

    1. **Parallel Mediators**: Multiple mediators operating simultaneously but independently
    2. **Serial Mediators**: Mediators operating in a causal chain (X → M1 → M2 → Y)
    3. **Moderated Mediation**: Mediation processes that vary across levels of a moderator
    """)

	st.markdown('<div class="section">Competitive vs. Complementary Mediation</div>', unsafe_allow_html=True)

	st.markdown("""
    **Complementary Mediation**:
    - Direct and indirect effects both exist and point in the same direction
    - Both effects are either positive or negative

    **Competitive Mediation**:
    - Direct and indirect effects point in opposite directions
    - One effect is positive while the other is negative
    - This can lead to suppression of the total effect

    <div class="tip">
    <strong>Example of Competitive Mediation:</strong> Exercise (X) might directly reduce anxiety (Y) (negative direct effect), but it might also increase physical discomfort (M), which could increase anxiety (positive indirect effect). The total effect might be smaller than the direct effect due to this competition.
    </div>
    """, unsafe_allow_html=True)

	st.markdown('<div class="section">Suppression vs. Mediation</div>', unsafe_allow_html=True)

	st.markdown("""
    **Suppression** occurs when:
    - Including the third variable (M) increases the magnitude of the relationship between X and Y
    - This is in contrast to mediation, where including M typically reduces the X-Y relationship

    **Key Difference**:
    - In mediation, the indirect and direct effects typically work in the same direction
    - In suppression, they work in opposite directions, revealing a stronger relationship when the suppressor is included
    """)

elif page == "Statistical Models":
	st.markdown('<div class="subtitle">Statistical Approaches to Mediation Analysis</div>', unsafe_allow_html=True)

	st.markdown('<div class="section">The Causal Steps Approach (Baron & Kenny)</div>', unsafe_allow_html=True)

	st.markdown("""
    The traditional approach to mediation analysis involves four steps:

    <div class="highlight">
    <strong>Step 1:</strong> Show that X significantly predicts Y (establish the total effect, path c)<br>
    <strong>Step 2:</strong> Show that X significantly predicts M (establish path a)<br>
    <strong>Step 3:</strong> Show that M significantly predicts Y when controlling for X (establish path b)<br>
    <strong>Step 4:</strong> Examine whether the effect of X on Y decreases substantially when M is included in the model
    </div>
    """, unsafe_allow_html=True)

	st.markdown('<div class="formula-box">', unsafe_allow_html=True)
	st.latex(r'''
    \begin{align}
    \text{Step 1: } Y &= i_1 + cX + e_1 \\
    \text{Step 2: } M &= i_2 + aX + e_2 \\
    \text{Step 3-4: } Y &= i_3 + c'X + bM + e_3
    \end{align}
    ''')
	st.markdown('</div>', unsafe_allow_html=True)

	st.markdown("""
    <div class="warning">
    <strong>Limitations of the Causal Steps Approach:</strong>
    <ul>
        <li>Low statistical power</li>
        <li>Does not directly test the significance of the indirect effect</li>
        <li>The first step (significant total effect) is no longer considered necessary</li>
        <li>Cannot handle multiple mediators efficiently</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

	st.markdown('<div class="section">Product of Coefficients Approach</div>', unsafe_allow_html=True)

	st.markdown("""
    This approach directly tests the indirect effect by calculating the product of paths a and b:

    1. Estimate path a (the effect of X on M)
    2. Estimate path b (the effect of M on Y, controlling for X)
    3. Calculate the indirect effect as a×b
    4. Test the significance of a×b
    """)

	st.markdown('<div class="formula-box">', unsafe_allow_html=True)
	st.latex(r'''
    \begin{align}
    \text{Indirect Effect} &= a \times b \\
    \text{Standard Error} &= \sqrt{a^2 \times SE_b^2 + b^2 \times SE_a^2}
    \end{align}
    ''')
	st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<div class="section">Bootstrap Methods</div>', unsafe_allow_html=True)

	st.markdown("""
    Bootstrap methods are widely recommended for testing indirect effects:

    1. **Non-parametric bootstrapping**: Resamples the data with replacement to create multiple samples
    2. For each bootstrap sample, calculates the indirect effect (a×b)
    3. Creates an empirical distribution of the indirect effect
    4. Constructs confidence intervals from this distribution
    5. If the confidence interval does not include zero, the indirect effect is considered significant

    <div class="tip">
    <strong>Advantages of Bootstrapping:</strong>
    <ul>
        <li>Does not assume normality of the indirect effect</li>
        <li>Higher statistical power than traditional approaches</li>
        <li>Works well with smaller sample sizes</li>
        <li>Can handle complex models with multiple mediators</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

	# Sample bootstrap distribution visualization
	bootstrap_fig, _ = create_bootstrap_plot(example_data)
	st.plotly_chart(bootstrap_fig, use_container_width=True)

	st.markdown('<div class="section">Sobel Test</div>', unsafe_allow_html=True)

	st.markdown("""
    The Sobel test is a parametric approach to test the significance of the indirect effect:

    1. Calculates the indirect effect as a×b
    2. Estimates the standard error of the indirect effect
    3. Computes a z-statistic and corresponding p-value
    """)

	st.markdown('<div class="formula-box">', unsafe_allow_html=True)
	st.latex(r'''
    \begin{align}
    z &= \frac{a \times b}{\sqrt{a^2 \times SE_b^2 + b^2 \times SE_a^2}} \\
    p &= 2 \times (1 - \Phi(|z|))
    \end{align}
    ''')
	st.markdown('</div>', unsafe_allow_html=True)

	st.markdown("""
    <div class="warning">
    <strong>Limitations of the Sobel Test:</strong>
    <ul>
        <li>Assumes that the indirect effect is normally distributed</li>
        <li>This assumption is often violated, especially in small samples</li>
        <li>Generally has lower power than bootstrap methods</li>
        <li>Not recommended for most applications</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

elif page == "Implementation with Pingouin":
	st.markdown('<div class="subtitle">Implementing Mediation Analysis with Pingouin</div>', unsafe_allow_html=True)

	st.markdown('<div class="section">The Pingouin Package</div>', unsafe_allow_html=True)

	st.markdown("""
    <div class="highlight">
    Pingouin is a statistical package in Python that provides a user-friendly API for conducting statistical analyses, including mediation analysis. It implements the bootstrap method for estimating indirect effects and their confidence intervals.
    </div>
    """, unsafe_allow_html=True)

	st.markdown("""
    Key features of Pingouin for mediation analysis:

    1. Easy-to-use function for performing mediation analysis
    2. Supports continuous and binary mediators
    3. Handles multiple parallel mediators
    4. Allows for inclusion of covariates
    5. Provides bootstrap confidence intervals for indirect effects
    6. Returns comprehensive statistics for all paths
    """)

	st.markdown('<div class="section">Basic Syntax</div>', unsafe_allow_html=True)

	st.code("""
    import pingouin as pg

    # Simple mediation analysis
    result = pg.mediation_analysis(
        data=df,      # DataFrame containing the variables
        x='X',        # Predictor variable (column name)
        m='M',        # Mediator variable (column name)
        y='Y',        # Outcome variable (column name)
        n_boot=1000,  # Number of bootstrap samples
        seed=42       # For reproducibility
    )

    print(result)
    """, language="python")

	st.markdown('<div class="section">Understanding the Output</div>', unsafe_allow_html=True)

	# Get example output
	result_df = pg.mediation_analysis(
		data=example_data,
		x='X',
		m='M',
		y='Y',
		n_boot=1000,
		seed=42
	)

	st.dataframe(result_df.style.highlight_max(axis=0, color='#e8f5e9').highlight_min(axis=0, color='#fff8e1'))

	st.markdown("""
    The output table includes:

    - **path**: The specific relationship being estimated
    - **coef**: The regression coefficient (effect size)
    - **se**: Standard error of the coefficient
    - **pval**: Two-sided p-value
    - **CI[2.5%]** and **CI[97.5%]**: Lower and upper bounds of the 95% confidence interval
    - **sig**: Whether the effect is statistically significant (Yes/No)

    Key rows:

    - **M ~ X**: Path a (effect of X on M)
    - **Y ~ M**: Path b (effect of M on Y, controlling for X)
    - **Total**: Total effect of X on Y (path c)
    - **Direct**: Direct effect of X on Y, controlling for M (path c')
    - **Indirect**: Indirect effect of X on Y through M (a×b)
    """)

	st.markdown('<div class="section">Advanced Options</div>', unsafe_allow_html=True)

	st.markdown("**Including Covariates**")

	st.code("""
    # Mediation analysis with covariates
    result = pg.mediation_analysis(
        data=df,
        x='X',
        m='M',
        y='Y',
        covar=['Z1', 'Z2'],  # Covariates to include in all regressions
        n_boot=1000,
        seed=42
    )
    """, language="python")

	st.markdown("**Multiple Parallel Mediators**")

	st.code("""
    # Mediation analysis with multiple mediators
    result = pg.mediation_analysis(
        data=df,
        x='X',
        m=['M1', 'M2', 'M3'],  # List of mediator variables
        y='Y',
        n_boot=1000,
        seed=42
    )
    """, language="python")

	st.markdown("**Binary Mediators**")

	st.code("""
    # Mediation analysis with a binary mediator
    result = pg.mediation_analysis(
        data=df,
        x='X',
        m='Mbin',  # Binary mediator (0/1)
        y='Y',
        n_boot=1000,
        seed=42
    )
    """, language="python")

	st.markdown("**Accessing Bootstrap Distribution**")

	st.code("""
    # Get bootstrap samples of the indirect effect
    result, dist = pg.mediation_analysis(
        data=df,
        x='X',
        m='M',
        y='Y',
        n_boot=1000,
        return_dist=True,  # Return bootstrap distribution
        seed=42
    )

    # Plot the distribution
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.histplot(dist, kde=True)
    plt.axvline(x=result.loc[result['path'] == 'Indirect', 'coef'].values[0], 
                color='r', linestyle='--')
    plt.xlabel('Indirect Effect')
    plt.title('Bootstrap Distribution of Indirect Effect')
    plt.show()
    """, language="python")

elif page == "Interactive Example":
	st.markdown('<div class="subtitle">Interactive Mediation Analysis Example</div>', unsafe_allow_html=True)

	st.markdown('<div class="section">Explore Relationships in the Data</div>', unsafe_allow_html=True)

	# Display the first few rows of the example data
	st.dataframe(example_data.head())

	# Create scatter plots of relationships
	relationship_fig = create_relationship_plots(example_data)
	st.plotly_chart(relationship_fig, use_container_width=True)

	st.markdown('<div class="section">Customize Your Mediation Analysis</div>', unsafe_allow_html=True)

	col1, col2 = st.columns(2)

	with col1:
		# Options for analysis
		x_var = st.selectbox("Select predictor (X):", ["X"])
		m_var = st.selectbox("Select mediator (M):", ["M", "Mbin"])
		y_var = st.selectbox("Select outcome (Y):", ["Y", "Ybin"])
		include_covar = st.checkbox("Include covariates")

		if include_covar:
			covar_options = [col for col in example_data.columns if col not in [x_var, m_var, y_var]]
			selected_covars = st.multiselect("Select covariates:", covar_options)
		else:
			selected_covars = None

		n_boots = st.slider("Number of bootstrap samples:", min_value=100, max_value=2000, value=1000, step=100)

	with col2:
		st.markdown("""
        <div class="highlight">
        <strong>Analysis Settings:</strong><br>
        Configure your mediation analysis by selecting the predictor, mediator, and outcome variables. You can also include covariates and adjust the number of bootstrap samples.
        </div>

        <div class="warning">
        <strong>Note:</strong> The outcome variable should be continuous for valid results. Binary outcomes are not fully supported by the current implementation.
        </div>
        """, unsafe_allow_html=True)

	# Run analysis button
	run_analysis = st.button("Run Mediation Analysis")

	if run_analysis:
		with st.spinner("Running mediation analysis..."):
			try:
				# Run the mediation analysis
				if y_var == "Ybin":
					st.warning("Note: Binary outcome variables may not provide valid results in this implementation.")

				if include_covar and selected_covars:
					result, dist = pg.mediation_analysis(
						data=example_data,
						x=x_var,
						m=m_var,
						y=y_var,
						covar=selected_covars,
						n_boot=n_boots,
						return_dist=True,
						seed=42
					)
				else:
					result, dist = pg.mediation_analysis(
						data=example_data,
						x=x_var,
						m=m_var,
						y=y_var,
						n_boot=n_boots,
						return_dist=True,
						seed=42
					)

				# Display results
				st.markdown('<div class="section">Results</div>', unsafe_allow_html=True)
				st.dataframe(result.style.highlight_max(axis=0, color='#e8f5e9').highlight_min(axis=0, color='#fff8e1'))

				# Create visualization of results
				col1, col2 = st.columns(2)

				with col1:
					# Create mediation diagram with results
					a_effect = result.loc[result['path'].str.contains(" ~ " + x_var), 'coef'].values[0]
					b_effect = result.loc[result['path'].str.contains(y_var + " ~ " + m_var), 'coef'].values[0]
					c_effect = result.loc[result['path'] == 'Total', 'coef'].values[0]
					c_prime = result.loc[result['path'] == 'Direct', 'coef'].values[0]
					indirect = result.loc[result['path'] == 'Indirect', 'coef'].values[0]

					fig = create_mediation_diagram(
						indirect_effect=round(indirect, 3),
						direct_effect=round(c_prime, 3),
						xm_effect=round(a_effect, 3),
						my_effect=round(b_effect, 3)
					)
					st.plotly_chart(fig, use_container_width=True)

				with col2:
					# Create bootstrap distribution plot
					fig = px.histogram(
						x=dist,
						nbins=30,
						histnorm='probability density',
						title='Bootstrap Distribution of Indirect Effect',
						labels={'x': 'Indirect Effect', 'y': 'Density'},
						color_discrete_sequence=['#673AB7']
					)

					# Add vertical line for the indirect effect
					indirect_effect = result.loc[result['path'] == 'Indirect', 'coef'].values[0]
					ci_lower = result.loc[result['path'] == 'Indirect', 'CI[2.5%]'].values[0]
					ci_upper = result.loc[result['path'] == 'Indirect', 'CI[97.5%]'].values[0]

					fig.add_vline(x=indirect_effect, line_width=2, line_dash="solid", line_color="#FF5722")
					fig.add_vline(x=ci_lower, line_width=2, line_dash="dash", line_color="#FFA000")
					fig.add_vline(x=ci_upper, line_width=2, line_dash="dash", line_color="#FFA000")

					# Add annotations
					fig.add_annotation(
						x=indirect_effect,
						y=0.5,
						text=f"Indirect Effect: {indirect_effect:.3f}",
						showarrow=True,
						arrowhead=1,
						ax=30,
						ay=-30,
						font=dict(color="#FF5722", size=12)
					)

					fig.add_annotation(
						x=(ci_lower + ci_upper) / 2,
						y=0.2,
						text=f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]",
						showarrow=False,
						font=dict(color="#FFA000", size=12)
					)

					# Update layout
					fig.update_layout(
						height=350,
						xaxis_title="Indirect Effect",
						yaxis_title="Density",
						template="plotly_white"
					)

					st.plotly_chart(fig, use_container_width=True)

				# Interpretation
				st.markdown('<div class="section">Interpretation</div>', unsafe_allow_html=True)

				# Determine type of mediation
				is_a_sig = result.loc[result['path'].str.contains(" ~ " + x_var), 'sig'].values[0] == 'Yes'
				is_b_sig = result.loc[result['path'].str.contains(y_var + " ~ " + m_var), 'sig'].values[0] == 'Yes'
				is_c_sig = result.loc[result['path'] == 'Total', 'sig'].values[0] == 'Yes'
				is_cprime_sig = result.loc[result['path'] == 'Direct', 'sig'].values[0] == 'Yes'
				is_indirect_sig = result.loc[result['path'] == 'Indirect', 'sig'].values[0] == 'Yes'

				if is_indirect_sig:
					if is_cprime_sig:
						mediation_type = "partial mediation"
					else:
						mediation_type = "complete mediation"
				else:
					mediation_type = "no mediation"

				# Sign of effects
				a_sign = "positive" if a_effect > 0 else "negative"
				b_sign = "positive" if b_effect > 0 else "negative"
				direct_sign = "positive" if c_prime > 0 else "negative"
				indirect_sign = "positive" if indirect > 0 else "negative"

				# Competitive or complementary
				if is_cprime_sig and is_indirect_sig:
					if (c_prime > 0 and indirect > 0) or (c_prime < 0 and indirect < 0):
						effect_type = "complementary"
					else:
						effect_type = "competitive"
				else:
					effect_type = "not applicable"

				# Create interpretation text
				interpretation = f"""
                <div class="highlight">
                <strong>Summary of Results:</strong><br>

                1. <strong>Path a ({x_var} → {m_var}):</strong> {a_sign.capitalize()} and {"statistically significant" if is_a_sig else "not statistically significant"} (β = {a_effect:.3f}).<br>

                2. <strong>Path b ({m_var} → {y_var}):</strong> {b_sign.capitalize()} and {"statistically significant" if is_b_sig else "not statistically significant"} (β = {b_effect:.3f}).<br>

                3. <strong>Direct effect ({x_var} → {y_var}):</strong> {direct_sign.capitalize()} and {"statistically significant" if is_cprime_sig else "not statistically significant"} (β = {c_prime:.3f}).<br>

                4. <strong>Indirect effect ({x_var} → {m_var} → {y_var}):</strong> {indirect_sign.capitalize()} and {"statistically significant" if is_indirect_sig else "not statistically significant"} (β = {indirect:.3f}, 95% CI [{ci_lower:.3f}, {ci_upper:.3f}]).<br>

                5. <strong>Total effect:</strong> {"Statistically significant" if is_c_sig else "Not statistically significant"} (β = {c_effect:.3f}).<br>

                <strong>Conclusion:</strong> The results suggest {mediation_type.upper()}. 
                """

				if mediation_type != "no mediation":
					interpretation += f"This is a {effect_type} mediation, where the direct and indirect effects are {direct_sign} and {indirect_sign}, respectively."

				interpretation += "</div>"

				st.markdown(interpretation, unsafe_allow_html=True)

				# Code to reproduce
				st.markdown('<div class="section">Code to Reproduce This Analysis</div>', unsafe_allow_html=True)

				code = f"""
                import pandas as pd
                import pingouin as pg
                import plotly.express as px

                # Run the mediation analysis
                result, dist = pg.mediation_analysis(
                    data=df,
                    x='{x_var}',
                    m='{m_var}',
                    y='{y_var}',
                    {"covar=" + str(selected_covars) + "," if include_covar and selected_covars else ""}
                    n_boot={n_boots},
                    return_dist=True,
                    seed=42
                )

                # Display results
                print(result)

                # Plot bootstrap distribution
                fig = px.histogram(
                    x=dist, 
                    nbins=30,
                    histnorm='probability density',
                    title='Bootstrap Distribution of Indirect Effect',
                    labels={{'x': 'Indirect Effect', 'y': 'Density'}}
                )
                fig.show()
                """

				st.code(code, language="python")

			except Exception as e:
				st.error(f"An error occurred during the analysis: {str(e)}")

	st.markdown('<div class="section">Generate Your Own Data</div>', unsafe_allow_html=True)

	st.markdown("""
    You can also simulate your own data with specific effect sizes to explore how different relationships affect mediation results.
    """)

	col1, col2, col3 = st.columns(3)

	with col1:
		st.markdown("**X → M effect (a)**")
		a_effect_sim = st.slider("Effect size:", min_value=-1.0, max_value=1.0, value=0.5, step=0.1)

	with col2:
		st.markdown("**M → Y effect (b)**")
		b_effect_sim = st.slider("Effect size:", min_value=-1.0, max_value=1.0, value=0.6, step=0.1)

	with col3:
		st.markdown("**X → Y direct effect (c')**")
		c_prime_sim = st.slider("Effect size:", min_value=-1.0, max_value=1.0, value=0.2, step=0.1)

	n_samples = st.slider("Sample size:", min_value=50, max_value=1000, value=200, step=50)

	generate_data = st.button("Generate Data and Run Analysis")

	if generate_data:
		with st.spinner("Generating data and running analysis..."):
			# Generate custom data
			np.random.seed(42)
			X_sim = np.random.normal(0, 1, n_samples)
			M_sim = a_effect_sim * X_sim + np.random.normal(0, np.sqrt(1 - a_effect_sim ** 2), n_samples)
			Y_sim = c_prime_sim * X_sim + b_effect_sim * M_sim + np.random.normal(0, np.sqrt(
				1 - (c_prime_sim ** 2 + b_effect_sim ** 2 + 2 * c_prime_sim * b_effect_sim * a_effect_sim)), n_samples)

			# Create dataframe
			sim_data = pd.DataFrame({
				'X': X_sim,
				'M': M_sim,
				'Y': Y_sim
			})

			# Run mediation analysis
			result_sim, dist_sim = pg.mediation_analysis(
				data=sim_data,
				x='X',
				m='M',
				y='Y',
				n_boot=1000,
				return_dist=True,
				seed=42
			)

			# Display results
			st.markdown('<div class="section">Simulation Results</div>', unsafe_allow_html=True)
			st.dataframe(result_sim.style.highlight_max(axis=0, color='#e8f5e9').highlight_min(axis=0, color='#fff8e1'))

			# Create visualization of results
			col1, col2 = st.columns(2)

			with col1:
				# Create mediation diagram with results
				a_effect_result = result_sim.loc[result_sim['path'] == 'M ~ X', 'coef'].values[0]
				b_effect_result = result_sim.loc[result_sim['path'] == 'Y ~ M', 'coef'].values[0]
				c_effect_result = result_sim.loc[result_sim['path'] == 'Total', 'coef'].values[0]
				c_prime_result = result_sim.loc[result_sim['path'] == 'Direct', 'coef'].values[0]
				indirect_result = result_sim.loc[result_sim['path'] == 'Indirect', 'coef'].values[0]

				fig = create_mediation_diagram(
					indirect_effect=round(indirect_result, 3),
					direct_effect=round(c_prime_result, 3),
					xm_effect=round(a_effect_result, 3),
					my_effect=round(b_effect_result, 3)
				)
				st.plotly_chart(fig, use_container_width=True)

			with col2:
				# Create bootstrap distribution plot
				fig = px.histogram(
					x=dist_sim,
					nbins=30,
					histnorm='probability density',
					title='Bootstrap Distribution of Indirect Effect',
					labels={'x': 'Indirect Effect', 'y': 'Density'},
					color_discrete_sequence=['#673AB7']
				)

				# Add vertical line for the indirect effect
				ci_lower = result_sim.loc[result_sim['path'] == 'Indirect', 'CI[2.5%]'].values[0]
				ci_upper = result_sim.loc[result_sim['path'] == 'Indirect', 'CI[97.5%]'].values[0]

				fig.add_vline(x=indirect_result, line_width=2, line_dash="solid", line_color="#FF5722")
				fig.add_vline(x=ci_lower, line_width=2, line_dash="dash", line_color="#FFA000")
				fig.add_vline(x=ci_upper, line_width=2, line_dash="dash", line_color="#FFA000")

				fig.add_annotation(
					x=indirect_result,
					y=0.5,
					text=f"Indirect Effect: {indirect_result:.3f}",
					showarrow=True,
					arrowhead=1,
					ax=30,
					ay=-30,
					font=dict(color="#FF5722", size=12)
				)

				st.plotly_chart(fig, use_container_width=True)

			# Compare expected vs. observed effects
			st.markdown('<div class="section">Expected vs. Observed Effects</div>', unsafe_allow_html=True)

			expected_indirect = a_effect_sim * b_effect_sim
			expected_total = c_prime_sim + expected_indirect

			comparison_df = pd.DataFrame({
				'Effect': ['X → M (a)', 'M → Y (b)', 'X → Y direct (c\')', 'X → Y indirect (a×b)', 'X → Y total (c)'],
				'Expected': [a_effect_sim, b_effect_sim, c_prime_sim, expected_indirect, expected_total],
				'Observed': [a_effect_result, b_effect_result, c_prime_result, indirect_result, c_effect_result],
				'Difference': [
					a_effect_result - a_effect_sim,
					b_effect_result - b_effect_sim,
					c_prime_result - c_prime_sim,
					indirect_result - expected_indirect,
					c_effect_result - expected_total
				]
			})

			st.dataframe(comparison_df.style.format({
				'Expected': '{:.3f}',
				'Observed': '{:.3f}',
				'Difference': '{:.3f}'
			}).background_gradient(subset=['Difference'], cmap='RdYlGn_r'))

elif page == "Interpretation":
	st.markdown('<div class="subtitle">Interpreting Mediation Analysis Results</div>', unsafe_allow_html=True)

	st.markdown('<div class="section">Key Statistics to Consider</div>', unsafe_allow_html=True)

	st.markdown("""
    When interpreting mediation analysis results, focus on these key statistics:

    1. **Indirect Effect (a×b)**
       - The magnitude of the effect that passes through the mediator
       - Statistical significance determined by confidence intervals
       - If the 95% CI does not include zero, the indirect effect is significant

    2. **Direct Effect (c')**
       - The effect of X on Y that is independent of the mediator
       - Compared to the total effect to determine the degree of mediation

    3. **Total Effect (c)**
       - The overall relationship between X and Y
       - Should equal the sum of direct and indirect effects

    4. **Path Coefficients (a and b)**
       - The individual relationships between X→M and M→Y
       - Both should be significant for mediation to occur (traditional view)
    """)

	st.markdown('<div class="section">Types of Mediation Outcomes</div>', unsafe_allow_html=True)

	col1, col2 = st.columns(2)

	with col1:
		st.markdown("""
        **Complete Mediation**
        - Direct effect (c') becomes non-significant
        - Indirect effect (a×b) is significant
        - The mediator fully explains the X-Y relationship

        **Partial Mediation**
        - Both direct and indirect effects are significant
        - The mediator explains part of the X-Y relationship

        **No Mediation**
        - Indirect effect is not significant
        - The mediator does not explain the X-Y relationship
        """)

	with col2:
		st.markdown("""
        **Complementary Mediation**
        - Direct and indirect effects have the same sign
        - Both effects work in the same direction

        **Competitive Mediation**
        - Direct and indirect effects have opposite signs
        - Effects work in opposite directions
        - Can lead to suppression of the total effect

        **Indirect-Only Mediation**
        - Indirect effect is significant, but direct effect is not
        - Similar to complete mediation
        """)

	st.markdown('<div class="section">Common Interpretation Pitfalls</div>', unsafe_allow_html=True)

	st.markdown("""
    <div class="warning">
    <strong>Beware of these common pitfalls when interpreting mediation results:</strong>

    1. <strong>Causal claims without experimental design</strong> - Mediation analysis alone cannot establish causality

    2. <strong>Ignoring temporal precedence</strong> - Ensure that variables occur in the expected order (X → M → Y)

    3. <strong>Focusing only on significance</strong> - Effect sizes are equally important for interpretation

    4. <strong>Overlooking assumptions</strong> - Check for violations of statistical assumptions

    5. <strong>Reverse causality</strong> - Consider alternative explanations for observed relationships

    6. <strong>Omitted variables</strong> - Unmeasured confounders can bias mediation estimates

    7. <strong>Insisting on significant total effect</strong> - Mediation can exist even without a significant total effect
    </div>
    """, unsafe_allow_html=True)

	st.markdown('<div class="section">Step-by-Step Interpretation Guide</div>', unsafe_allow_html=True)

	st.markdown("""
    Follow these steps when interpreting mediation analysis results:

    1. **Examine the total effect (c)**
       - Is there an overall relationship between X and Y?
       - Note that a significant total effect is no longer considered necessary for mediation

    2. **Check the indirect effect (a×b)**
       - Is it statistically significant? (CI does not include zero)
       - What is the magnitude of the effect?

    3. **Assess the direct effect (c')**
       - Is it statistically significant?
       - How does it compare to the total effect?

    4. **Determine the type of mediation**
       - Complete, partial, or no mediation?
       - Complementary or competitive?

    5. **Consider practical significance**
       - What is the proportion mediated? (indirect effect / total effect)
       - Is the magnitude of the indirect effect meaningful in your context?

    6. **Check for alternative explanations**
       - Could other variables explain the observed relationships?
       - Consider theoretical perspectives and previous research
    """)

	st.markdown('<div class="section">Example Interpretation</div>', unsafe_allow_html=True)

	# Run a sample analysis for interpretation
	result = pg.mediation_analysis(
		data=example_data,
		x='X',
		m='M',
		y='Y',
		n_boot=1000,
		seed=42
	)

	st.dataframe(result.style.highlight_max(axis=0, color='#e8f5e9').highlight_min(axis=0, color='#fff8e1'))

	st.markdown("""
    <div class="highlight">
    <strong>Example Interpretation:</strong>

    The mediation analysis examined whether M mediates the relationship between X and Y. Results show:

    1. <strong>Path a (X → M):</strong> Significant positive effect (β = 0.495, p < 0.001), indicating that increases in X are associated with increases in M.

    2. <strong>Path b (M → Y):</strong> Significant positive effect (β = 0.596, p < 0.001), indicating that increases in M are associated with increases in Y when controlling for X.

    3. <strong>Indirect effect (X → M → Y):</strong> Significant positive effect (β = 0.295, 95% CI [0.217, 0.383]), suggesting that M significantly mediates the relationship between X and Y.

    4. <strong>Direct effect (X → Y):</strong> Significant positive effect (β = 0.274, p < 0.001), indicating that X still has a direct effect on Y when controlling for M.

    5. <strong>Total effect (X → Y):</strong> Significant positive effect (β = 0.569, p < 0.001), representing the overall relationship between X and Y.

    <strong>Conclusion:</strong> The results suggest partial mediation, as both the direct and indirect effects are significant. This is a complementary mediation, as both the direct and indirect effects are positive. Approximately 51.8% of the total effect is mediated through M (0.295/0.569 = 0.518).
    </div>
    """, unsafe_allow_html=True)

elif page == "References":
	st.markdown('<div class="subtitle">References and Further Reading</div>', unsafe_allow_html=True)

	st.markdown('<div class="section">Key References</div>', unsafe_allow_html=True)

	st.markdown("""
    1. Baron, R. M., & Kenny, D. A. (1986). The moderator-mediator variable distinction in social psychological research: Conceptual, strategic, and statistical considerations. *Journal of Personality and Social Psychology, 51*(6), 1173-1182.

    2. Hayes, A. F. (2018). *Introduction to mediation, moderation, and conditional process analysis: A regression-based approach* (2nd ed.). Guilford Press.

    3. MacKinnon, D. P., Fairchild, A. J., & Fritz, M. S. (2007). Mediation analysis. *Annual Review of Psychology, 58*, 593-614.

    4. Preacher, K. J., & Hayes, A. F. (2008). Asymptotic and resampling strategies for assessing and comparing indirect effects in multiple mediator models. *Behavior Research Methods, 40*(3), 879-891.

    5. Zhao, X., Lynch Jr, J. G., & Chen, Q. (2010). Reconsidering Baron and Kenny: Myths and truths about mediation analysis. *Journal of Consumer Research, 37*(2), 197-206.

    6. Fiedler, K., Schott, M., & Meiser, T. (2011). What mediation analysis can (not) do. *Journal of Experimental Social Psychology, 47*(6), 1231-1236.
    """)

	st.markdown('<div class="section">Python Packages for Mediation Analysis</div>', unsafe_allow_html=True)

	st.markdown("""
    1. **Pingouin** ([Documentation](https://pingouin-stats.org/build/html/generated/pingouin.mediation_analysis.html))
       - User-friendly API for mediation analysis
       - Supports bootstrap methods for indirect effects
       - Handles continuous and binary mediators

    2. **statsmodels**
       - General-purpose statistical package
       - Can be used to implement mediation analysis manually

    3. **PyProcessMacro**
       - Python implementation of Hayes' PROCESS macro
       - Supports complex mediation and moderation models
    """)

	st.markdown('<div class="section">Recommended Readings for Beginners</div>', unsafe_allow_html=True)

	st.markdown("""
    1. Hayes, A. F. (2022). *Introduction to mediation, moderation, and conditional process analysis: A regression-based approach* (3rd ed.). Guilford Press.
       - Comprehensive introduction to mediation analysis
       - Accessible for beginners with practical examples

    2. MacKinnon, D. P. (2008). *Introduction to statistical mediation analysis*. Routledge.
       - Detailed coverage of statistical methods for mediation
       - Includes advanced topics and applications

    3. Shrout, P. E., & Bolger, N. (2002). Mediation in experimental and nonexperimental studies: New procedures and recommendations. *Psychological Methods, 7*(4), 422-445.
       - Influential paper on bootstrap methods for mediation
       - Clear explanation of key concepts
    """)

	st.markdown('<div class="section">Online Resources</div>', unsafe_allow_html=True)

	st.markdown("""
    1. **Introduction to Mediation Analysis** - University of Virginia Library
       - [https://data.library.virginia.edu/introduction-to-mediation-analysis/](https://data.library.virginia.edu/introduction-to-mediation-analysis/)

    2. **PROCESS macro for SPSS and SAS** - Andrew F. Hayes
       - [https://www.processmacro.org/](https://www.processmacro.org/)

    3. **Mediation Analysis Using Python** - Towards Data Science
       - [https://towardsdatascience.com/mediation-analysis-using-python-d13058783ae7](https://towardsdatascience.com/mediation-analysis-using-python-d13058783ae7)
    """)

# Footer
st.markdown("---")
st.markdown("Created by Dr. Merwan Roudane | Understanding Mediation Analysis | 2025")
