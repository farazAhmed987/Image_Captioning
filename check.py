import tensorflow as tf
import os
import h5py
import zipfile

MODEL_PATH = 'model_checkpoints/final_caption_model.keras'

def check_file_integrity():
    """Check if the file is corrupted or has wrong format"""
    print("=== File Integrity Check ===")
    
    # Check if it's a valid HDF5 file
    try:
        with h5py.File(MODEL_PATH, 'r') as f:
            print("✅ File is a valid HDF5 format")
            print(f"Keys in file: {list(f.keys())}")
            return True
    except Exception as e:
        print(f"❌ Not a valid HDF5 file: {e}")
    
    # Check if it's a ZIP file (Keras files are actually ZIP archives)
    try:
        with zipfile.ZipFile(MODEL_PATH, 'r') as z:
            print("✅ File is a valid ZIP archive")
            print(f"Files in archive: {z.namelist()[:10]}...")  # Show first 10 files
            return True
    except Exception as e:
        print(f"❌ Not a valid ZIP file: {e}")
    
    # Check file header
    try:
        with open(MODEL_PATH, 'rb') as f:
            header = f.read(8)
            print(f"File header (hex): {header.hex()}")
            print(f"File header (ascii): {header}")
    except Exception as e:
        print(f"❌ Cannot read file header: {e}")
    
    return False

def try_different_loading_methods():
    """Try various methods to load the model"""
    print("\n=== Trying Different Loading Methods ===")
    
    methods = [
        ("Standard load_model", lambda: tf.keras.models.load_model(MODEL_PATH)),
        ("Load with compile=False", lambda: tf.keras.models.load_model(MODEL_PATH, compile=False)),
        ("Load with custom_objects", lambda: tf.keras.models.load_model(MODEL_PATH, custom_objects={})),
        ("Load as H5", lambda: tf.keras.models.load_model(MODEL_PATH.replace('.keras', '.h5') if os.path.exists(MODEL_PATH.replace('.keras', '.h5')) else MODEL_PATH)),
    ]
    
    for method_name, method_func in methods:
        try:
            print(f"\nTrying: {method_name}")
            model = method_func()
            print(f"✅ SUCCESS with {method_name}!")
            print(f"Model type: {type(model)}")
            if hasattr(model, 'summary'):
                try:
                    model.summary()
                except:
                    print("Model loaded but summary failed")
            return model
        except Exception as e:
            print(f"❌ {method_name} failed: {str(e)[:100]}...")
    
    return None

def create_backup_and_fix():
    """Create backup and try to fix the file"""
    print("\n=== Creating Backup and Attempting Fix ===")
    
    # Create backup
    backup_path = MODEL_PATH + '.backup'
    try:
        import shutil
        shutil.copy2(MODEL_PATH, backup_path)
        print(f"✅ Backup created: {backup_path}")
    except Exception as e:
        print(f"❌ Backup failed: {e}")
        return False
    
    # Try to fix by renaming extension
    h5_path = MODEL_PATH.replace('.keras', '.h5')
    try:
        shutil.copy2(MODEL_PATH, h5_path)
        print(f"✅ Created H5 copy: {h5_path}")
        
        # Try loading the H5 version
        model = tf.keras.models.load_model(h5_path)
        print("✅ H5 version loads successfully!")
        return model
    except Exception as e:
        print(f"❌ H5 fix failed: {e}")
    
    return None

def main():
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Model path: {MODEL_PATH}")
    print(f"File exists: {os.path.exists(MODEL_PATH)}")
    print(f"File size: {os.path.getsize(MODEL_PATH)} bytes")
    
    # Step 1: Check file integrity
    is_valid = check_file_integrity()
    
    # Step 2: Try different loading methods
    model = try_different_loading_methods()
    
    if model is None:
        print("\n=== All standard methods failed ===")
        # Step 3: Try backup and fix
        model = create_backup_and_fix()
    
    if model is not None:
        print("\n🎉 MODEL LOADED SUCCESSFULLY!")
        print("You can now update your app.py to use the working method.")
    else:
        print("\n💔 ALL METHODS FAILED")
        print("Recommendations:")
        print("1. The model file is likely corrupted")
        print("2. Try re-downloading or regenerating the model")
        print("3. Check if you have other model files (.h5, folder format)")
        print("4. Consider retraining the model")

if __name__ == "__main__":
    main()