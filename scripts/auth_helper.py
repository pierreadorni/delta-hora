#!/usr/bin/env python3
"""
Helper script for Hugging Face authentication.
"""

def check_auth_status():
    """Check if user is authenticated with Hugging Face."""
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"✅ Authenticated as: {user_info['name']}")
        return True
    except Exception as e:
        print(f"❌ Not authenticated: {e}")
        return False

def login_interactive():
    """Interactive login to Hugging Face."""
    try:
        from huggingface_hub import login
        print("Starting interactive login...")
        login()
        print("✅ Login successful!")
        return True
    except Exception as e:
        print(f"❌ Login failed: {e}")
        return False

def main():
    print("Hugging Face Authentication Helper")
    print("=" * 40)
    print()
    
    # Check current status
    print("Checking authentication status...")
    if check_auth_status():
        print("\nYou're already authenticated! You can use Llama models.")
        print("Use: python scripts/train.py --config configs/llama2_config.json")
        return
    
    print("\nTo use Llama models, you need to authenticate.")
    print("Steps:")
    print("1. Get access at: https://huggingface.co/meta-llama/Llama-2-7b-hf")
    print("2. Create token at: https://huggingface.co/settings/tokens")
    print("3. Run authentication below")
    print()
    
    choice = input("Would you like to login now? (y/n): ").lower().strip()
    
    if choice in ['y', 'yes']:
        if login_interactive():
            print("\n✅ Authentication successful!")
            print("You can now use: python scripts/train.py --config configs/llama2_config.json")
        else:
            print("\n❌ Authentication failed. Try manual login:")
            print("   huggingface-cli login")
    else:
        print("\nAlternatively:")
        print("• Command line: huggingface-cli login")
        print("• Environment: export HUGGINGFACE_HUB_TOKEN=your_token")
        print("• Use DialoGPT: python scripts/train.py --config configs/default_config.json")

if __name__ == "__main__":
    main()
