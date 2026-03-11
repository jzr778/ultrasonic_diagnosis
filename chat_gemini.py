import base64
import os
from openai import OpenAI
from PIL import Image
import io

import config


class InteractiveGeminiChat:
    def __init__(self):
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=config.VLM_API_KEY,
            base_url=config.VLM_BASE_URL,
        )

        # 对话历史（保持上下文）
        self.conversation_history = []

        # 当前会话的附件（文本文件+图片）
        self.current_attachments = {
            'text_files': [],
            'image_files': []
        }

    def add_attachments(self, user_input):
        """解析用户输入，添加附件"""
        parts = user_input.strip().split()
        new_text_files = []
        new_image_files = []
        question_parts = []

        i = 0
        while i < len(parts):
            part = parts[i]

            # 检查是否是文件
            if os.path.exists(part):
                # 根据扩展名判断文件类型
                ext = os.path.splitext(part.lower())[1]
                if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
                    new_image_files.append(part)
                    print(f"📷 已添加图片: {os.path.basename(part)}")
                elif ext in ['.txt', '.md', '.py', '.json', '.html', '.xml', '.csv']:
                    new_text_files.append(part)
                    print(f"📄 已添加文本文件: {os.path.basename(part)}")
                else:
                    # 未知类型，当作文本文件尝试
                    new_text_files.append(part)
                    print(f"📄 已添加文件: {os.path.basename(part)}")
            else:
                question_parts.append(part)
            i += 1

        # 更新附件
        self.current_attachments['text_files'].extend(new_text_files)
        self.current_attachments['image_files'].extend(new_image_files)

        # 返回纯文本问题
        question = ' '.join(question_parts)
        return question

    def clear_attachments(self):
        """清空当前附件"""
        self.current_attachments = {
            'text_files': [],
            'image_files': []
        }
        print("🗑️  已清空所有附件")

    def read_text_file(self, file_path):
        """读取文本文件内容"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    return f.read()
            except Exception as e:
                return f"❌ 无法读取文件: {e}"
        except Exception as e:
            return f"❌ 读取文件出错: {e}"

    def encode_image(self, image_path):
        """编码图片为base64"""
        try:
            with Image.open(image_path) as img:
                # 优化大小
                max_size = 1024
                if max(img.size) > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

                # 转换为base64
                buffered = io.BytesIO()
                img_format = img.format if img.format else 'JPEG'
                img.save(buffered, format=img_format, quality=85)
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

                mime_type = f"image/{img_format.lower()}" if img_format else "image/jpeg"

                return {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{img_base64}"
                    }
                }
        except Exception as e:
            print(f"❌ 图片处理失败: {e}")
            return None

    def prepare_message_content(self, question):
        """准备消息内容（包含所有附件和当前问题）"""
        content = []

        # 首先添加所有附件的内容
        all_text = []

        # 添加文本文件内容
        for text_file in self.current_attachments['text_files']:
            if os.path.exists(text_file):
                file_content = self.read_text_file(text_file)
                if not file_content.startswith("❌"):
                    filename = os.path.basename(text_file)
                    all_text.append(f"【附件: {filename}】")
                    all_text.append(file_content)
                    all_text.append("")  # 空行

        # 如果有附件，添加说明
        if self.current_attachments['text_files'] or self.current_attachments['image_files']:
            if all_text:
                content.append({
                    "type": "text",
                    "text": "\n".join(all_text)
                })

        # 添加图片附件
        for image_file in self.current_attachments['image_files']:
            if os.path.exists(image_file):
                encoded_image = self.encode_image(image_file)
                if encoded_image:
                    content.append(encoded_image)

        # 最后添加当前问题
        if question:
            if content:  # 如果有附件，添加分隔符
                content.append({
                    "type": "text",
                    "text": f"\n--- 当前问题 ---\n{question}"
                })
            else:  # 如果没有附件，直接添加问题
                content.append({
                    "type": "text",
                    "text": question
                })

        return content

    def chat_round(self, user_input):
        """单轮对话"""
        # 解析用户输入，添加附件
        question = self.add_attachments(user_input)

        if not question and not self.current_attachments['text_files'] and not self.current_attachments['image_files']:
            return "❌ 请输入内容或添加文件"

        print(f"\n📤 准备发送...")

        # 准备消息内容
        message_content = self.prepare_message_content(question)

        # 添加到对话历史
        self.conversation_history.append({
            "role": "user",
            "content": message_content
        })

        # 选择模型（根据是否有图片）
        has_images = any(item.get("type") == "image_url" for item in message_content)
        model = "gemini-3-pro-preview"

        if not model:
            return "❌ 没有可用的模型"

        print(f"🔄 使用模型: {model}")

        try:
            # 发送请求（包含完整对话历史）
            response = self.client.chat.completions.create(
                model=model,
                messages=self.conversation_history,
                temperature=0.1,
            )

            assistant_reply = response.choices[0].message.content

            # 添加到对话历史
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_reply
            })

            return assistant_reply

        except Exception as e:
            error_msg = str(e)
            if "model_not_found" in error_msg:
                return f"❌ 模型不支持，请尝试其他模型。错误: {error_msg}"
            else:
                return f"❌ API错误: {error_msg}"

    def select_model(self, has_images):
        """选择模型"""
        model = "gemini-3-pro-preview"

        # 简单测试
        self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            temperature=0.2,
        )

        return None

    def show_conversation(self):
        """显示当前对话"""
        print("\n📜 对话历史:")
        print("=" * 60)

        if not self.conversation_history:
            print("  对话为空")
            return

        for i, msg in enumerate(self.conversation_history):
            role = "👤 用户" if msg["role"] == "user" else "🤖 助手"

            # 提取文本预览
            text_preview = ""
            if isinstance(msg["content"], str):
                text_preview = msg["content"]
            elif isinstance(msg["content"], list):
                text_items = []
                for item in msg["content"]:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_items.append(item.get("text", ""))

                text_preview = " | ".join(text_items)

            # 截断预览
            preview = text_preview[:100] + "..." if len(text_preview) > 100 else text_preview

            print(f"{i + 1:2d}. {role}: {preview}")

        print("=" * 60)

    def show_attachments(self):
        """显示当前附件"""
        print("\n📎 当前附件:")
        print("-" * 40)

        if self.current_attachments['text_files']:
            print("📄 文本文件:")
            for tf in self.current_attachments['text_files']:
                print(f"  • {os.path.basename(tf)}")

        if self.current_attachments['image_files']:
            print("📷 图片文件:")
            for img in self.current_attachments['image_files']:
                print(f"  • {os.path.basename(img)}")

        if not self.current_attachments['text_files'] and not self.current_attachments['image_files']:
            print("  暂无附件")

        print("-" * 40)

    def interactive_chat(self):
        """交互式聊天主循环"""
        print("=" * 70)
        print("🤖 Gemini交互式对话系统")
        print("=" * 70)
        print("\n🎯 特点:")
        print("  • 真正的对话模式（保持上下文）")
        print("  • 附件会一直保留，直到手动清除")
        print("  • 支持多轮基于附件的对话")
        print("\n📝 使用方法:")
        print("  1. 添加文件: prompt.txt image1.jpg image2.png")
        print("  2. 问问题: 请分析这些附件")
        print("  3. 继续对话: 那么具体来说...")
        print("  4. 添加更多文件: new_image.jpg")
        print("  5. 继续提问: 结合新文件再分析")
        print("\n🔧 可用命令:")
        print("  /clear    - 清空附件")
        print("  /history  - 查看对话历史")
        print("  /attach   - 查看当前附件")
        print("  /reset    - 重置对话（清空所有）")
        print("  /exit     - 退出")
        print("=" * 70)

        while True:
            try:
                print(f"\n💬 当前对话轮次: {len(self.conversation_history) // 2}")
                self.show_attachments()

                user_input = input("\n👤 您: ").strip()

                if not user_input:
                    continue

                # 处理命令
                if user_input.lower() == '/exit':
                    print("👋 再见！")
                    break

                elif user_input.lower() == '/clear':
                    self.clear_attachments()
                    continue

                elif user_input.lower() == '/history':
                    self.show_conversation()
                    continue

                elif user_input.lower() == '/attach':
                    self.show_attachments()
                    continue

                elif user_input.lower() == '/reset':
                    self.conversation_history = []
                    self.clear_attachments()
                    print("🔄 对话已重置")
                    continue

                print("🤖 思考中...")

                # 进行对话
                response = self.chat_round(user_input)

                print(f"\n{'=' * 70}")
                print("💡 Gemini回复:")
                print(f"{'=' * 70}")
                print(response)
                print(f"{'=' * 70}")

            except KeyboardInterrupt:
                print("\n\n👋 对话已中断")
                break
            except Exception as e:
                print(f"\n❌ 错误: {e}")


def main():
    """主函数"""
    print("🚀 启动交互式Gemini对话系统...")

    # 检查文件
    print("当前目录:", os.getcwd())
    print("目录内容:", [f for f in os.listdir('.') if os.path.isfile(f)][:10])

    # 启动聊天
    chat = InteractiveGeminiChat()
    chat.interactive_chat()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ 程序启动失败: {e}")
        print("\n请确保:")
        print("1. 已安装依赖: pip install openai pillow")
        print("2. API密钥有效")
        print("3. 网络连接正常")

# prompt.txt panoramic_1.jpg panoramic_2.jpg panoramic_3.jpg panoramic_4.jpg
# prompt_avm.txt avm.jpg