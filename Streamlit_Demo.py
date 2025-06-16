import streamlit as st

# Set page configuration first, before any other Streamlit commands
st.set_page_config(
    page_title="ArabVLM | منصة الذكاء الاصطناعي لوصف الصور",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

from PIL import Image
import torch
import os
from pathlib import Path
import time
import logging
from vllm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from vllm.conversation import conv_templates, SeparatorStyle
from vllm.model.builder import load_pretrained_model
from vllm.utils import disable_torch_init
from vllm.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === App Configuration ===
TITLE = "ArabVLM | منصة الذكاء الاصطناعي لوصف الصور"
LOGO_PATH = "assets/logo.png"  # Update with actual logo path


MODEL_PATH = '/BigData-KSU/ArabVLM'
#################################################################################################
MODEL_BASE = 'ALLaM-AI/ALLaM-7B-Instruct-preview'
###################################################################################################


SAMPLE_IMAGES_PATH = Path("/media/pc/e/2025/ArabVLM/sample_images")
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Custom CSS for professional styling with improved colors
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700&family=Poppins:wght@300;400;500;600&display=swap');

    :root {
        --primary-color: #0a5d87;
        --primary-light: #e6f3f8;
        --secondary-color: #28a745;
        --secondary-light: #e6f7e9;
        --accent-color: #ffc107;
        --text-color: #343a40;
        --light-text: #6c757d;
        --border-color: #dee2e6;
        --bg-color: #f8f9fa;
        --card-bg: #ffffff;
        --header-bg: linear-gradient(135deg, #0a5d87 0%, #1a8cbe 100%);
    }

    html, body, [class*="css"] {
        font-family: 'Tajawal', 'Poppins', sans-serif;
        direction: rtl;
        color: var(--text-color);
    }

    /* Main app background */
    .stApp {
        background-color: var(--bg-color);
    }

    /* Header styling */
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
        background: var(--header-bg);
        padding: 20px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .main-header h1, .main-header h2, .main-header h3 {
        color: white;
    }

    /* Make tabs more prominent */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1px;
        background-color: var(--bg-color);
        border-radius: 10px;
        padding: 0px 10px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        border-radius: 5px 5px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: var(--primary-color);
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background-color: white !important;
        border-top: 3px solid var(--primary-color);
        color: var(--primary-color) !important;
        font-weight: 600;
    }

    /* Card styling for descriptions */
    .description-box {
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 20px;
        background-color: var(--card-bg);
        margin-top: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }

    .description-box h3 {
        color: var(--primary-color);
        border-bottom: 2px solid var(--primary-light);
        padding-bottom: 10px;
        margin-bottom: 15px;
    }

    /* Category title styling */
    .category-title {
        background-color: var(--primary-color);
        color: white;
        padding: 10px 15px;
        border-radius: 5px;
        margin-top: 30px;
        margin-bottom: 15px;
        font-weight: 500;
    }

    /* Image card styling */
    .image-card {
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 15px;
        transition: all 0.3s ease;
        overflow: hidden;
        background-color: var(--card-bg);
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }

    .image-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        border-color: var(--primary-color);
    }

    /* Button styling */
    .stButton button {
        width: 100%;
        border-radius: 5px;
        background-color: var(--primary-color);
        color: white;
        margin-top: 5px;
        margin-bottom: 10px;
        padding: 2px 5px;
        font-size: 0.9rem;
        border: none;
        transition: all 0.2s ease;
        font-family: 'Tajawal', sans-serif;
    }

    .stButton button:hover {
        background-color: #08486a;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }

    /* Special buttons */
    button[data-testid="baseButton-secondary"] {
        background-color: var(--secondary-color) !important;
    }

    button[data-testid="baseButton-secondary"]:hover {
        background-color: #218838 !important;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f0f5f8;
        border-right: 1px solid var(--border-color);
    }

    [data-testid="stSidebar"] h3 {
        color: var(--primary-color);
        border-bottom: 2px solid var(--primary-light);
        padding-bottom: 5px;
        font-weight: 500;
    }

    /* File uploader styling */
    [data-testid="stFileUploader"] {
        background-color: var(--primary-light);
        padding: 10px;
        border-radius: 8px;
        border: 1px dashed var(--primary-color);
    }

    /* Footer styling */
    .footer {
        text-align: center;
        margin-top: 50px;
        padding: 20px;
        border-top: 1px solid var(--border-color);
        font-size: 0.9rem;
        color: var(--light-text);
        background-color: #f0f5f8;
        border-radius: 0 0 10px 10px;
    }

    /* Separator styling */
    hr {
        border-top: 1px solid var(--border-color);
        margin: 30px 0;
    }

    /* Info box styling */
    .stAlert {
        border-radius: 8px;
        border-left-width: 5px !important;
    }

    /* For mobile responsiveness */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.5rem;
        }
        .main-header h2 {
            font-size: 1.2rem;
        }
        .main-header h3 {
            font-size: 1rem;
        }
    }
</style>
"""


# === UI Helper Functions ===
def display_header():
    """Display application header with logo and Saudi flag"""
    st.markdown('<div class="main-header">', unsafe_allow_html=True)

    # Three-column layout for header: Flag, Title, Logo
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        # Saudi Arabia flag
        saudi_flag_html = """
        <div style="text-align: center; margin-top: 20px;">
            <img src="https://upload.wikimedia.org/wikipedia/commons/0/0d/Flag_of_Saudi_Arabia.svg" 
                 width="100" alt="علم المملكة العربية السعودية" 
                 style="border: 1px solid rgba(255,255,255,0.3); border-radius: 4px;">
        </div>
        """
        st.markdown(saudi_flag_html, unsafe_allow_html=True)

    with col2:
        # Application title only (removed company name)
        st.markdown("""
            <div style="text-align: center;">
                <h1 style="margin-top: 5px; font-weight: 600; color: white; text-shadow: 1px 1px 3px rgba(0,0,0,0.3);">
                    ArabVLM
                </h1>
                <h2 style="font-size: 1.5rem; margin-top: 5px; font-weight: 400; color: rgba(255,255,255,0.9); text-shadow: 1px 1px 2px rgba(0,0,0,0.2);">
                    منصة الذكاء الاصطناعي لوصف الصور
                </h2>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        # If logo exists, display it
        try:
            if os.path.exists(LOGO_PATH):
                st.image(LOGO_PATH, width=100, use_container_width=False)
            else:
                # Simple placeholder for logo (without DH initials)
                st.markdown("""
                    <div style="text-align: center; margin-top: 20px;">
                        <div style="display: inline-block; width: 90px; height: 90px; background: linear-gradient(135deg, #0a5d87 0%, #1a8cbe 100%); 
                                    border-radius: 50%; color: white; display: flex; align-items: center; justify-content: center;
                                    border: 3px solid rgba(255,255,255,0.8); box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
                            <span style="font-size: 36px; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">🔍</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        except:
            pass

    st.markdown("</div>", unsafe_allow_html=True)

    # Description under header (removed DataHarmony mention)
    st.markdown("""
        <div style="text-align: center; background-color: #f0f5f8; padding: 15px; border-radius: 0 0 10px 10px; margin-top: -16px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
            <p style="font-size: 1.1rem; margin: 0; color: var(--primary-color);">
                <i style="margin-left: 8px; font-size: 1.2rem;">&#128065;</i>
                منصة متطورة لوصف الصور باستخدام نموذج ArabVLM، المدرب خصيصاً للغة العربية
            </p>
        </div>
    """, unsafe_allow_html=True)


def display_footer():
    """Display application footer with minimal branding"""
    st.components.v1.html("""
<div style="text-align: center; margin-top: 30px;">
    <div style="display: inline-flex; justify-content: center; align-items: center; background: linear-gradient(135deg, #0a5d87 0%, #1a8cbe 100%); padding: 10px 20px; border-radius: 30px; gap: 15px;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/0/0d/Flag_of_Saudi_Arabia.svg" 
             width="30" alt="علم المملكة العربية السعودية" 
             style="border: 1px solid rgba(255,255,255,0.3); border-radius: 3px;">
        <span style="font-weight: 600; color: white; font-size: 1.1rem; text-shadow: 1px 1px 1px rgba(0,0,0,0.2);">
            ArabVLM
        </span>
    </div>

    <p style="color: #666; margin-top: 20px;">تم تطوير هذا النموذج لدعم اللغة العربية في مجال الذكاء الاصطناعي البصري</p>
    <p style="color: #0a5d87; font-weight: 500;">© 2025 جميع الحقوق محفوظة</p>

    <div style="display: flex; justify-content: center; gap: 30px; margin-top: 15px; font-size: 0.9rem;">
        <a href="#" style="text-decoration: none; color: #0a5d87;">الشروط والأحكام</a>
        <a href="#" style="text-decoration: none; color: #0a5d87;">سياسة الخصوصية</a>
        <a href="#" style="text-decoration: none; color: #0a5d87;">اتصل بنا</a>
    </div>
</div>
""")


def display_simple_image_grid():
    """Display all sample images in a simple direct grid layout"""
    logger.info(f"Loading images from path: {SAMPLE_IMAGES_PATH}")

    if not SAMPLE_IMAGES_PATH.exists():
        st.warning("مجلد الصور التوضيحية غير متوفر")
        return

    # Direct approach: display all image files from immediate subfolders
    categories = []
    for item in SAMPLE_IMAGES_PATH.iterdir():
        if item.is_dir():
            categories.append(item)

    if not categories:
        st.warning("لم يتم العثور على مجلدات تحتوي على صور")
        return

    # Log what we found
    logger.info(f"Found {len(categories)} categories: {[c.name for c in categories]}")

    # Display images from each category
    for category in categories:
        # Better category title with icon and styling
        st.markdown(f"""
            <div class="category-title">
                <i style="margin-left: 8px;">&#128194;</i> {category.name}
            </div>
        """, unsafe_allow_html=True)

        # Get all image files in this category
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.gif',"*.webp"]:
            image_files.extend(list(category.glob(ext)))

        logger.info(f"Found {len(image_files)} images in category {category.name}")

        if not image_files:
            st.info(f"لا توجد صور في هذه الفئة")
            continue

        # Create a grid layout using columns
        num_cols = 4  # Number of columns in the grid
        cols = st.columns(num_cols)

        # Display each image in the grid
        for i, img_path in enumerate(image_files):
            with cols[i % num_cols]:
                try:
                    img = Image.open(img_path)

                    # Add container with shadow and styling
                    st.markdown(f"""
                        <div class="image-card">
                            <div style="font-size: 0.8rem; color: var(--light-text); margin-bottom: 5px; text-align: center;">
                                {img_path.name}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    # Show the image
                    st.image(img, use_container_width=True)

                    # Button to describe this image
                    if st.button("وصف الصورة", key=f"btn_{category.name}_{i}"):
                        st.session_state.selected_image = img_path
                        if 'sample_description' in st.session_state:
                            del st.session_state.sample_description
                        st.rerun()
                except Exception as e:
                    st.error(f"خطأ في عرض الصورة {img_path.name}: {e}")

        # Add a separator between categories
        st.markdown("<hr>", unsafe_allow_html=True)


def get_image_base64(image_path, size=None):
    """Convert image to base64 for embedded display, with optional resizing"""
    import base64
    from io import BytesIO

    try:
        img = Image.open(image_path)

        # Resize image if size is specified
        if size:
            img.thumbnail(size, Image.LANCZOS)

        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=70)  # Reduced quality for faster loading
        return base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        logger.error(f"Error converting image to base64: {e}")
        return ""


# === Model Loading Function ===
@st.cache_resource
def load_model():
    try:
        logger.info(f"Loading model from {MODEL_PATH}")
        start_time = time.time()

        disable_torch_init()
        model_path = os.path.abspath(MODEL_PATH)
        model_name = get_model_name_from_path(model_path)

        tokenizer, model, processor, context_len = load_pretrained_model(
            model_path, MODEL_BASE, model_name, device=DEVICE
        )

        logger.info(f"Model loaded successfully in {time.time() - start_time:.2f} seconds")
        return tokenizer, model, processor, context_len
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        st.error(f"فشل تحميل النموذج: {str(e)}")
        return None, None, None, None


# === Image Processing Functions ===
def describe_image_with_model(image: Image.Image, prompt="وصف الصورة بالتفصيل"):
    """Generate a description for the provided image using ArabVLM model"""
    try:
        if model is None:
            return "لم يتم تحميل النموذج بشكل صحيح. يرجى التحقق من الإعدادات."

        # Process image
        image_processor = processor['image']
        conv = conv_templates['llava_llama_2'].copy()

        # Convert image to tensor
        image_tensor = image_processor.preprocess(image.convert("RGB"), return_tensors='pt')['pixel_values']
        tensor = image_tensor.to(model.device, dtype=torch.float16)

        # Prepare prompt
        cur_prompt = prompt
        cur_prompt = DEFAULT_IMAGE_TOKEN + '\n' + cur_prompt
        conv.append_message(conv.roles[0], cur_prompt)
        conv.append_message(conv.roles[1], None)
        formatted_prompt = conv.get_prompt()

        # Tokenize input
        input_ids = tokenizer_image_token(
            formatted_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(model.device)

        # Configure stopping criteria
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

        # Generate response
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=tensor,
                do_sample=False,
                max_new_tokens=2048,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )

        # Decode and return response
        response = tokenizer.decode(output_ids[0, input_ids.shape[1]:])
        return response.strip()

    except Exception as e:
        logger.error(f"Error describing image: {str(e)}")
        return f"حدث خطأ أثناء معالجة الصورة: {str(e)}"


# === Main Application Function ===
def main():
    # Page config is already set at the top of the script

    # Inject custom CSS
    st.markdown(custom_css, unsafe_allow_html=True)

    # Display header
    display_header()

    # Sidebar content
    with st.sidebar:
        # Add minimal branding to sidebar (DataHarmony mention kept here only)
        st.markdown("""
            <div style="text-align: center; margin-bottom: 20px;">
                <div style="display: inline-flex; align-items: center; background: linear-gradient(135deg, #0a5d87 0%, #1a8cbe 100%); 
                            padding: 10px 15px; border-radius: 10px; margin-bottom: 10px;">
                    <span style="color: white; font-weight: 600; font-size: 1rem; text-shadow: 1px 1px 1px rgba(0,0,0,0.2);">
                        تناغم البيانات
                    </span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("### عن المنصة")
        st.info("""
            منصة ArabVLM هي منصة متخصصة في وصف الصور باللغة العربية باستخدام تقنيات الذكاء الاصطناعي المتقدمة.

            النموذج مدرب على مجموعة واسعة من الصور والأوصاف العربية لضمان دقة الوصف وطبيعية اللغة.
        """)

        st.markdown("### إعدادات")
        model_status = "✅ جاهز للاستخدام" if model is not None else "❌ غير محمل"
        st.write(f"حالة النموذج: {model_status}")

        if st.button("إعادة تحميل النموذج"):
            st.cache_resource.clear()
            st.rerun()

        # Add advanced options in sidebar
        st.markdown("### خيارات متقدمة")
        custom_prompt = st.text_area(
            "نص الوصف الافتراضي",
            value="وصف الصورة بالتفصيل",
            help="سيتم استخدام هذا النص للطلب من النموذج وصف الصور"
        )
        st.session_state.default_prompt = custom_prompt

    # Two tabs: Upload and Samples
    tab1, tab2 = st.tabs(["📤 رفع صورة", "🖼️ مكتبة الصور"])

    # Tab 1: Upload image
    with tab1:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### رفع صورة من جهازك")
            uploaded_file = st.file_uploader(
                "اختر صورة",
                type=["jpg", "jpeg", "png","webp"],
                help="يمكنك رفع صور بصيغة JPG أو PNG"
            )

        if uploaded_file:
            with col2:
                # Process uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="الصورة المرفوعة", use_container_width=True)

            if st.button("وصف الصورة المرفوعة", key="describe_uploaded", use_container_width=True):
                with st.spinner("جاري توليد الوصف..."):
                    description = describe_image_with_model(image, prompt=st.session_state.default_prompt)

                # Display result
                st.markdown("<div class='description-box'>", unsafe_allow_html=True)
                st.markdown("### وصف الصورة")
                st.write(description)
                st.markdown("</div>", unsafe_allow_html=True)

                # Feedback and export options
                col1a, col2a = st.columns(2)
                with col1a:
                    st.button("👍 وصف ممتاز", key="thumb_up_uploaded")
                with col2a:
                    st.button("👎 وصف ضعيف", key="thumb_down_uploaded")

                st.download_button(
                    label="💾 حفظ النتيجة",
                    data=f"الصورة: {uploaded_file.name}\n\nالوصف:\n{description}",
                    file_name=f"وصف_{uploaded_file.name.split('.')[0]}.txt",
                    mime="text/plain"
                )

    # Tab 2: Sample image library
    with tab2:
        st.markdown("### مكتبة الصور المصنفة")
        st.markdown("اختر من الفئات أدناه لوصف الصور باستخدام نموذج ArabVLM")

        # Simple direct display of images by category
        display_simple_image_grid()

    # Results container for sample images
    if 'selected_image' in st.session_state:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("## نتيجة وصف الصورة المختارة")

        img_path = st.session_state.selected_image
        image = Image.open(img_path)

        # Two columns: Image on left, description on right
        col_img, col_desc = st.columns([1, 2])

        with col_img:
            st.image(image, caption=f"{os.path.basename(os.path.dirname(img_path))} / {img_path.name}",
                     use_container_width=True)

            if st.button("إعادة الوصف", key="redescribe_sample"):
                with st.spinner("جاري إعادة توليد الوصف..."):
                    st.session_state.sample_description = describe_image_with_model(
                        image, prompt=st.session_state.default_prompt
                    )
                    st.rerun()

        with col_desc:
            # Generate description if not already done
            if 'sample_description' not in st.session_state:
                with st.spinner("جاري توليد الوصف..."):
                    st.session_state.sample_description = describe_image_with_model(
                        image, prompt=st.session_state.default_prompt
                    )

            # Display result
            st.markdown("<div class='description-box'>", unsafe_allow_html=True)
            st.markdown("### وصف الصورة")
            st.write(st.session_state.sample_description)
            st.markdown("</div>", unsafe_allow_html=True)

            # Feedback and export options
            col1b, col2b = st.columns(2)
            with col1b:
                st.button("👍 وصف ممتاز", key="thumb_up_sample")
            with col2b:
                st.button("👎 وصف ضعيف", key="thumb_down_sample")

            st.download_button(
                label="💾 حفظ النتيجة",
                data=f"الصورة: {img_path.name}\n\nالوصف:\n{st.session_state.sample_description}",
                file_name=f"وصف_{img_path.stem}.txt",
                mime="text/plain"
            )

    # Footer
    display_footer()


# === Load model ===
tokenizer, model, processor, context_len = load_model()

# === Entry Point ===
if __name__ == "__main__":
    main()